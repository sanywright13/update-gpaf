"""Flower Server."""

from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple
#from MulticoreTSNE import print_function
import flwr
import mlflow
import base64
import pickle
import torch
from typing import List, Tuple, Optional, Dict, Callable, Union
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import json
from flwr.server.strategy import Strategy,FedAvg
from fedprox.models import Encoder, Classifier,test,test_gpaf ,reparameterize,sample_labels,generate_feature_representation,ServerDiscriminator,GlobalGenerator

from flwr.server.client_proxy import ClientProxy
from fedprox.features_visualization import extract_features_and_labels,StructuredFeatureVisualizer

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
import os
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

class GPAFStrategy(FedAvg):
    def __init__(
        self,
       experiment_name,
        num_classes: int=2,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 2,
            min_evaluate_clients : int =0,  # No clients for evaluation
   evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
  
    ) -> None:
        super().__init__()
        

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
         experiment_id = mlflow.create_experiment(experiment_name)
         #print(f"Created new experiment with ID: {experiment_id}")
         experiment = mlflow.get_experiment(experiment_id)
        else:
         print(f"Using existing experiment with ID: {experiment.experiment_id}")
    
        # Store MLflow reference
        self.mlflow = mlflow
        #experiment_id = mlflow.create_experiment(experiment_name)
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="server") as run:
         self.server_run_id = run.info.run_id
         # Log server parameters
         mlflow.log_params({
                "num_classes": num_classes,
                "min_fit_clients": min_fit_clients,
                "fraction_fit": fraction_fit
            })
         
        #on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        # Initialize the generator and its optimizer here
        self.num_classes =num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_avg_accuracy=0.0
        # Initialize the generator and its optimizer here
        self.num_classes = num_classes
        #self.latent_dim = 64
        #self.hidden_dim = 256  # Define hidden_dim
        #self.output_dim = 64   # Define output_dim
      
        # Adjust noise_dim to account for label space
        self.latent_dim = 64  # Changed from 64 to 192
        self.hidden_dim = 256
        self.output_dim = 64  # Changed from 64 to 192
        self.noise_dim = self.latent_dim - self.num_classes  # 192 - 2 = 190
        # Initialize the new generator
        # Initialize the new generator
        self.generator = GlobalGenerator(
            noise_dim=self.noise_dim ,
            label_dim=self.num_classes,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        ).to(self.device)
        
        # Initialize optimizer with learning rate scheduler
        self.optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        
     
        # Save generator state initialization
        self.generator_state = {
            k: v.cpu().numpy() 
            for k, v in self.generator.state_dict().items()
        }
        #initiliaze global descriminator 
        # Initialize server discriminator with GRL
        self.server_discriminator = ServerDiscriminator(
            feature_dim=self.latent_dim, 
            num_domains=self.num_classes
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        self.discriminator_optimizer = torch.optim.Adam(
            self.server_discriminator.parameters(), 
            lr=0.0002
        )
        # Initialize label_probs with a default uniform distribution
        self.label_probs = {label: 1.0 / self.num_classes for label in range(self.num_classes)}
        # Store client models for ensemble predictions
        self.client_classifiers = {}
        self.feature_visualizer =StructuredFeatureVisualizer(
        num_clients=3,  # total number of clients
        num_classes=self.num_classes,           # number of classes in your dataset

save_dir="feature_visualizations_gpaf"
          )
    def compute_generator_loss(self,ensemble_logits, labels, mu, logvar, kl_weight=0.01):
      """
      Compute the total loss for the generator:
      1. Ensemble prediction loss (cross-entropy)
      2. KL divergence to regularize the latent space
      """
      # Prediction loss
      #print(f'hdgdg logits server {ensemble_logits}')
      ce_loss = F.cross_entropy(ensemble_logits, labels)
      #kl_weight=0.01, 
      entropy_weight=1.0  # Increased weight
      # KL divergence loss
      kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
      # 3. Entropy components
      avg_probs = torch.softmax(ensemble_logits, dim=-1)
    
      # Individual entropy (minimize)
      individual_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=1).mean()
    
      # Batch entropy (maximize via minimizing negative entropy)
      batch_probs = avg_probs.mean(dim=0)
      batch_entropy = -torch.sum(batch_probs * torch.log(batch_probs + 1e-10))
    
      # Total entropy loss
      entropy_loss = individual_entropy - batch_entropy
      # Total loss with weighting
      total_loss = ce_loss + kl_weight * kl_loss 
    
      return total_loss, ce_loss, kl_loss
           
    def initialize_parameters(self, client_manager):
        print("=== Initializing Parameters ===")
        # Initialize your models
        encoder = Encoder()
        classifier = Classifier(num_features=encoder.num_features)
        
        # Get parameters in the correct format
        encoder_params = [val.cpu().numpy() for key, val in encoder.state_dict().items() if "num_batches_tracked" not in key]  
        classifier_params = [val.cpu().numpy() for _, val in classifier.state_dict().items()]
        
        # Combine parameters
        ndarrays = encoder_params + classifier_params
        parameters=ndarrays_to_parameters(ndarrays)
        #print("Parameter types:")
        '''
        for i, param in enumerate(ndarrays):
          print(f"  Param {i}: type={type(param)}, shape={param.shape if isinstance(param, np.ndarray) else 'N/A'}")
        '''        
        
        return parameters
        
    def get_generator_parameters(self) -> Parameters:
        """Convert generator parameters to Flower Parameters format."""
        generator_state_dict = self.generator.state_dict()
        # Convert state dict to numpy arrays
        generator_params = [
            param.cpu().numpy() 
            for param in generator_state_dict.values()
        ]
        return ndarrays_to_parameters(generator_params)
    def num_evaluate_clients(self, client_manager: ClientManager) -> Tuple[int, int]:
      """Return the sample size and required number of clients for evaluation."""
      num_clients = client_manager.num_available()
      return max(int(num_clients * self.fraction_evaluate), self.min_evaluate_clients), self.min_available_clients
    #1first run
    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate evaluation results."""
        if not results:
            return None, {}
       
        accuracies = {}
        self.current_features = {}
        self.current_labels = {}
      
        # Extract all accuracies from evaluation
        with self.mlflow.start_run(run_id=self.server_run_id):  

         accuracies = {}
         for client_proxy, eval_res in results:
            client_id = client_proxy.cid

            accuracy = eval_res.metrics.get("accuracy", 0.0)
            accuracies[f"client_{client_id}"] = accuracy
            metrics = eval_res.metrics
            # Get features and labels if available
            if "features" in metrics and "labels" in metrics:
              
              features_np = pickle.loads(base64.b64decode(metrics.get("features").encode('utf-8')))
              labels_np = pickle.loads(base64.b64decode(metrics.get("labels").encode('utf-8')))
              self.current_features[client_id] = features_np
              self.current_labels[client_id] = labels_np
            
            #print(f"Stored data for client {client_id}")
            # Log in a format that will show up as separate lines in MLflow
            self.mlflow.log_metrics({
                    f"accuracy_client_{client_id}": accuracy
                }, step=server_round)

                      
        # Calculate average accuracy
        avg_accuracy = sum(accuracies.values()) / len(accuracies)
        # Only visualize if we have all the data and accuracy improved
        if avg_accuracy > self.best_avg_accuracy:
          #print(f'==visualization===')
          self.best_avg_accuracy = avg_accuracy
          self.feature_visualizer.visualize_all_clients_by_class(
            features_dict=self.current_features,
            labels_dict=self.current_labels,
            accuracies=accuracies,
            epoch=server_round,
            stage="validation"
          )
          batch_size = 13 # Example batch size
          #noise_dim = self.latent_dim  # Noise dimension
          label_dim = self.num_classes # Label dimension
          # Generate global features with their conditioned labels
          noise = torch.randn(batch_size, self.noise_dim).to(self.device)
          labels = sample_labels(batch_size, self.label_probs)
          labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        
          with torch.no_grad():
            global_z = self.generator(noise, labels_one_hot).cpu().numpy()
          # Visualize with class-colored global features
          self.feature_visualizer.visualize_global_local_features(
            client_features_dict=self.current_features,
            global_z=global_z,
            client_labels_dict=self.current_labels,
            global_labels=labels.numpy(),
            epoch=server_round
        )
        return avg_accuracy, {"accuracy": avg_accuracy}
    def configure_evaluate(
      self, server_round: int, parameters: Parameters, client_manager: ClientManager
) -> List[Tuple[ClientProxy, EvaluateIns]]:
      
      """Configure the next round of evaluation."""
      #parameters=parameters_to_ndarrays(parameters)
      #print(f' ejkfzejrk {type(parameters)}')
      # Sample clients
      sample_size, min_num_clients = self.num_evaluate_clients(client_manager)
      clients = client_manager.sample(
        num_clients=sample_size, min_num_clients=min_num_clients
    )
      evaluate_config = {"server_round": server_round}  # Pass the round number in config
      # Create EvaluateIns for each client
      '''
      evaluate_config = (
        self.on_evaluate_config_fn(server_round) if self.on_evaluate_config_fn is not None else {}
      )
      '''
      print(f"Server sending round number: {server_round}")  # Debug print
      evaluate_ins = EvaluateIns(parameters, evaluate_config)
     
      # Return client-EvaluateIns pairs
      return [(client, evaluate_ins) for client in clients]   
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        print(f'===server evaluation=======')
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: flwr.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, flwr.common.FitIns]]:
      """Configure the next round of training and send current generator state."""
      """ send config variable to clients  and client recieve it in fit function"""
      # Generate z representation using the generator
      batch_size = 13 # Example batch size
      noise_dim = self.latent_dim  # Noise dimension
      label_dim = self.num_classes # Label dimension
      config={}
    
      mu = torch.zeros(batch_size, noise_dim)  # Mean of the Gaussian
      logvar = torch.zeros(batch_size, noise_dim)  # Log variance of the Gaussian
      noise = reparameterize(mu, logvar)  # Reparameterized noise
      # Sample labels and convert to one-hot encoding
      labels =sample_labels(batch_size, self.label_probs)
      labels_one_hot = F.one_hot(labels, num_classes=label_dim).float()
  
      generator_state = self.generator.state_dict()
        
      # Convert generator parameters to NumPy arrays
      generator_numpy_params = [param.cpu().detach().numpy() for param in generator_state.values()]
      
      # Serialize generator parameters into a JSON string
      generator_params_serialized = json.dumps([param.tolist() for param in generator_numpy_params])
      
      # 3. Get discriminator state
      discriminator_state = {
        k: v.cpu().numpy() 
        for k, v in self.server_discriminator.state_dict().items()
    }
      discriminator_params_serialized = json.dumps(
        {k: v.tolist() for k, v in discriminator_state.items()}
    )
      config = {
            "round": server_round,
            "generator_params": generator_params_serialized,
            "discriminator_state": discriminator_params_serialized
        }

      client_proxies = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_fit_clients,
        )
      
      # Create fit instructions with current parameters and config
      fit_ins = []
      for client in client_proxies:
            fit_ins.append((client, flwr.common.FitIns(parameters, config)))
         
      return fit_ins

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, flwr.common.FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate results and update generator."""
        #print(f'results faillure {failures}')    
        if not results:
            return None, {}

       
        # Generate z representation using the generator
        batch_size = 13 # Example batch size
        noise_dim = self.latent_dim  # Noise dimension
        label_dim = self.num_classes  # Label dimension
        # Aggregate label distributions
        global_label_counts = {}
        total_samples = 0
        # Extract features and weights from clients
        features_list = []
        weights_list = []
        domain_labels = []
        # Aggregate label counts
        accuracy_metrics = {}
        for domain_idx, (client_proxy, fit_res) in enumerate(results):
                # Parse label distribution from metrics
                label_distribution_str = fit_res.metrics.get("label_distribution", "{}")
                label_distribution = json.loads(label_distribution_str)
                client_num_samples = fit_res.num_examples
                accuracy = fit_res.metrics.get("accuracy", 0.0)
                # Get client's features
                features_serialized = fit_res.metrics["features"]
                client_features = pickle.loads(base64.b64decode(features_serialized.encode('utf-8')))
                client_features = torch.tensor(client_features).to(self.device)
                features_list.append(client_features)
                # Create domain labels (which client/domain these features come from)
              

                # Add domain labels for each sample in this client's batch
                domain_labels.extend([domain_idx] * client_features.size(0))                     
                
                weights_list.append(fit_res.parameters)
                # Accumulate label counts
                for label, prob in label_distribution.items():
                    label = int(label)
                    count = int(float(prob) * client_num_samples)
                    global_label_counts[label] = global_label_counts.get(label, 0) + count
                    total_samples += count
         
        # Compute global label probabilities
        if total_samples > 0:
            self.label_probs = {
                label: count / total_samples 
                for label, count in global_label_counts.items()
            }
        else:
            self.label_probs = {}
        
        #print(f'Label distribution: {self.label_probs}')
        # Extract num_encoder_params from the first client's metrics
        num_encoder_params = int(results[0][1].metrics["num_encoder_params"])
      
        encoder_params_list = []
        classifier_params_list = []
        num_samples_list = []
        
        for _, fit_res in results:
            # Convert parameters to NumPy arrays
            client_parameters = parameters_to_ndarrays(fit_res.parameters)
            
            # Validate parameter length
            if len(client_parameters) < num_encoder_params:
                print(f"Warning: Client parameters length ({len(client_parameters)}) is less than num_encoder_params ({num_encoder_params})")
                continue
            # Debugging: Print the shape of each parameter
        
            # Split parameters into encoder and classifier
            encoder_params = client_parameters[:num_encoder_params]
            classifier_params = client_parameters[num_encoder_params:]
            #print("Shapes of client parameters:")
           
            encoder_params_list.append(encoder_params)
            classifier_params_list.append(classifier_params)
            num_samples_list.append(fit_res.num_examples)
            for i, param in enumerate(classifier_params):
              print(f"  Parameter {i}: {param.shape}")
        # Train server discriminator
        all_features = torch.cat(features_list, dim=0)
        all_domain_labels = torch.tensor(domain_labels, dtype=torch.long, device=self.device)        
        # Multiple training iterations for discriminator
        '''
        for _ in range(10):
            self.discriminator_optimizer.zero_grad()
            
            # Forward pass through discriminator (includes GRL)
            domain_predictions = self.server_discriminator(all_features)
            
            # Compute discriminator loss
            d_loss = F.cross_entropy(domain_predictions, all_domain_labels)
            
            # Update discriminator
            d_loss.backward()
            print(f' discriminator loss {d_loss}')
            self.discriminator_optimizer.step()

        '''
        if not encoder_params_list or not classifier_params_list:
            print("No valid parameters to aggregate")
            return None, {}
        # Aggregate parameters using FedAvg
        aggregated_encoder_params = self._fedavg_parameters(encoder_params_list, num_samples_list)
        aggregated_classifier_params = self._fedavg_parameters(classifier_params_list, num_samples_list)
        
        # Combine aggregated parameters
        aggregated_params = aggregated_encoder_params + aggregated_classifier_params
        
   
        self._train_generator(self.generator,self.label_probs,classifier_params_list)
     
        with self.mlflow.start_run(run_id=self.server_run_id):  

            self.mlflow.log_metrics({
                "round": server_round,
                "num_clients": len(results),
                "num_failures": len(failures),
                "total_samples": sum(r.num_examples for _, r in results)
            }, step=server_round)

        # Prepare config for next round
        config = {
            "server_round": server_round,
            "discriminator_state": self.server_discriminator.state_dict()
        }
        # Clear memory
        del features_list
        del all_features
        del all_domain_labels
        
        return ndarrays_to_parameters(aggregated_params),config
        
    def _fedavg_parameters(
        self, params_list: List[List[np.ndarray]], num_samples_list: List[int]
    ) -> List[np.ndarray]:
        """Aggregate parameters using FedAvg (weighted averaging)."""
        if not params_list:
            return []

        print("==== aggregation===")
        total_samples = sum(num_samples_list)

        # Initialize aggregated parameters with zeros
        aggregated_params = [np.zeros_like(param) for param in params_list[0]]

        # Weighted sum of parameters
        for params, num_samples in zip(params_list, num_samples_list):
            for i, param in enumerate(params):
                aggregated_params[i] += param * num_samples

        # Weighted average of parameters
        aggregated_params = [param / total_samples for param in aggregated_params]

        return aggregated_params
   
    def _create_client_model(self, classifier_params: NDArrays) -> nn.Module: 
        # Initialize the classifier
        classifier = Classifier(num_features=self.latent_dim, num_classes=2).to(self.device)
      
        # Create state dict with proper device placement
        classifier_state_dict = OrderedDict()
        
        for (name, orig_param), param_data in zip(classifier.named_parameters(), classifier_params):
            # Convert numpy array to tensor if necessary
            if isinstance(param_data, np.ndarray):
                param_tensor = torch.tensor(param_data, dtype=orig_param.dtype)
            else:
                param_tensor = param_data.clone()
         
            # Move to correct device
            param_tensor = param_tensor.to(self.device)
            classifier_state_dict[name] = param_tensor
            #print(f"Successfully processed {name} with shape {param_tensor.shape}")
        
        # Load the state dict
        classifier.load_state_dict(classifier_state_dict, strict=True)
        
        return classifier
        
    

    def _train_generator(self,generator,label_probs, classifier_params:  List[NDArrays]):
      """Train the generator using the ensemble of client classifiers."""
      with self.mlflow.start_run(run_id=self.server_run_id):  
        
        label_dim = 2
        hidden_dim = 256
        
        batch_size = 13
        learning_rate = 0.001
        num_epochs = 10
        # Optimizer
        optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

       # Training loop
        for epoch in range(num_epochs):
          print('====== Training Generator=====')
          epoch_loss = 0.0
          total_ce_loss = 0
          total_kl_loss = 0
          total_loss=0
          num_batches = 0
          # Sample labels and prepare tensors
          labels = sample_labels(batch_size, label_probs)
          labels = torch.tensor(labels, dtype=torch.long).to(self.device)
          labels_one_hot = F.one_hot(labels, num_classes=label_dim).float().to(self.device)
        
         
          noise = torch.randn(batch_size, generator.noise_dim)
          # Generate features with reparameterization
          features, mu, logvar = generator(noise, labels_one_hot,return_distribution=True)
          #optimizer.zero_grad()
         
          # Get logits from client classifiers
          logits = []
          
          for params in classifier_params:
                # Convert parameters to proper format if they're named parameters
                if hasattr(params, '__iter__') and not isinstance(params, (list, tuple, np.ndarray)):
                    params = [p.detach().cpu().numpy() for _, p in params]
                
                client_model = self._create_client_model(params)
                client_model = client_model.to(self.device)
                client_model.eval()
                
                client_logits = client_model(features)
                logits.append(client_logits)
                    
          if not logits:
                raise ValueError("No valid logits generated from client models")
                
          # Average the logits from all clients
          avg_logits = torch.mean(torch.stack(logits), dim=0)
            
          # Compute loss
          #oss = criterion(avg_logits, labels)
          # Compute losses
          loss, ce_loss, kl_loss = self.compute_generator_loss(
                avg_logits, labels, mu, logvar
            )
            
            
          # Backpropagate and update generator
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
            
          #epoch_loss += loss.item()
          total_loss += loss.item()
          total_ce_loss += ce_loss.item()
          total_kl_loss += kl_loss.item()
          # Log metrics using mlflow directly
          self.mlflow.log_metrics({
                    "generator_loss": epoch_loss,
                    "epoch": epoch,
                }, step=epoch)

          print(f"Generator Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}")
            
        save_dir = "z_representations"
       