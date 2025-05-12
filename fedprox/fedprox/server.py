from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple
#from MulticoreTSNE import print_function
import flwr
import mlflow
from torch.cuda.amp import autocast, GradScaler
import base64
import pickle
from torch.distributions import Dirichlet, Categorical
import torch
from typing import List, Tuple, Optional, Dict, Callable, Union
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import json
from flwr.server.strategy import Strategy,FedAvg
from fedprox.models import Encoder, Classifier,test,test_gpaf ,GlobalGenerator,reparameterize,sample_labels,generate_feature_representation,LocalDiscriminator,GlobalDiscriminator
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.client_proxy import ClientProxy
from fedprox.features_visualization import extract_features_and_labels,StructuredFeatureVisualizer
import csv
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
        num_classes: int=9,
        fraction_fit: float = 1.0,
        fraction_evaluate=1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients=2,
        min_available_clients=2,
        batch_size=13,
        
   evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
  
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
      
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
         experiment_id = mlflow.create_experiment(experiment_name)
         print(f"Created new experiment with ID: {experiment_id}")
         experiment = mlflow.get_experiment(experiment_id)
        else:
         print(f"Using existing experiment with ID: {experiment.experiment_id}")
    
        # Store MLflow reference
        self.mlflow = mlflow
        self.client_to_domain={}
        self.num_domains = self.min_fit_clients
        self.batch_size=batch_size

        print(f'num domain : {self.min_fit_clients}')
       
        #experiment_id = mlflow.create_experiment(experiment_name)
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="server") as run:
         self.server_run_id = run.info.run_id
         # Log server parameters
         mlflow.log_params({
                "num_classes": num_classes,
                "min_fit_clients": min_fit_clients,
                "fraction_fit": fraction_fit
            })
         
        print(f"Created MLflow run for server: {self.server_run_id}")
        #on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        # Initialize the generator and its optimizer here
        self.num_classes =num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_avg_accuracy=0.0
        # Initialize the generator and its optimizer here
      
        self.latent_dim = 128
        self.hidden_dim = 256
        self.output_dim = 128   # Match latent_dim
        self.noise_dim = 64     # This can stay if you're happy with 64-dim noise
        #self.noise_dim = self.latent_dim - self.num_classes  # 192 - 2 = 190
        self.noise_dim=64
        self.domain_dim=32
        # Initialize the new generator
        self.generator = GlobalGenerator(
            noise_dim=self.noise_dim ,
            label_dim=self.num_classes,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
     

        ).to(self.device)
        # Initialize server discriminator with GRL
        self.global_model_dis=Classifier(latent_dim=64, num_classes=self.num_classes)
      
        # Save generator state initialization
        self.generator_state = {
            k: v.cpu().numpy() 
            for k, v in self.generator.state_dict().items()
        }
        self.discriminator = GlobalDiscriminator().to(self.device)

        
        '''
        self.local_discriminator = LocalDiscriminator(
            feature_dim=self.latent_dim, 
            num_domains=self.num_classes
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001)
      
        self.discriminator_optimizer = torch.optim.Adam(
            self.server_discriminator.parameters(), 
            lr=0.0002
        )
        '''
        # Initialize label_probs with a default uniform distribution
        self.label_probs = {label: 1.0 / self.num_classes for label in range(self.num_classes)}
        # Store client models for ensemble predictions
        self.client_classifiers = {}
        self.feature_visualizer =StructuredFeatureVisualizer(
        num_clients=2,  # total number of clients
        num_classes=self.num_classes,           # number of classes in your dataset

save_dir="feature_visualizations_gpaf"
         )
               
    def initialize_parameters(self, client_manager):
        print("=== Initializing Parameters ===")
        # Initialize your models
        #encoder = Encoder(self.latent_dim)
        #encoder =Encoder(self.latent_dim)
        latent_dim = 128  # or your desired latent dimension
        encoder =Encoder(self.latent_dim)
        classifier = Classifier(latent_dim=latent_dim, num_classes=self.num_classes)
        local_discriminator = LocalDiscriminator(
            feature_dim=self.latent_dim, 
            num_domains=self.num_domains
        ).to(self.device)
        
        # Get parameters in the correct format
        encoder_params = [val.cpu().numpy() for key, val in encoder.state_dict().items() if "num_batches_tracked" not in key]  
        classifier_params = [val.cpu().numpy() for _, val in classifier.state_dict().items()]
        discriminator_params = [val.cpu().numpy() for _, val in local_discriminator.state_dict().items()]

        # Combine parameters
        ndarrays = encoder_params + classifier_params + discriminator_params
        parameters=ndarrays_to_parameters(ndarrays)
        #print("Parameter types:")
        # Get client IDs from client_manager
        if not self.client_to_domain:
          # Retrieve all clients from the client_manager
          clients = client_manager.all()  # Returns a dict: {cid: ClientProxy}
          self.client_ids = list(clients.keys())  # Extract client IDs as a list
        
          # Ensure the number of clients matches num_domains (optional check)
          if len(self.client_ids) != self.num_domains:
            print(f"Warning: Number of clients ({len(self.client_ids)}) does not match num_domains ({self.num_domains}).")
            # Fallback to default IDs if needed
            self.client_ids = [f"client_{i}" for i in range(self.num_domains)]
        
          # Create client-to-domain mapping
          self.client_to_domain = {cid: idx for idx, cid in enumerate(self.client_ids)}
          print(f'client to ids {self.client_to_domain}')
             
        
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
        clients_ids=[]
        # Extract all accuracies from evaluation
        with self.mlflow.start_run(run_id=self.server_run_id):  

         accuracies = {}
         for client_proxy, eval_res in results:
            client_id = client_proxy.cid
            clients_ids.append(client_id)
            accuracy = eval_res.metrics.get("accuracy", 0.0)
            accuracies[f"client_{client_id}"] = accuracy
            metrics = eval_res.metrics
            # Get features and labels if available
            if "features" in metrics and "labels" in metrics:
              
              features_np = pickle.loads(base64.b64decode(metrics.get("features").encode('utf-8')))
              labels_np = pickle.loads(base64.b64decode(metrics.get("labels").encode('utf-8')))
              self.current_features[client_id] = features_np
              self.current_labels[client_id] = labels_np
            
            print(f"Stored data for client {client_id}")
           
            self.mlflow.log_metrics({
                    f"accuracy_client_{client_id}": accuracy
                }, step=server_round)

                      
        # Calculate average accuracy
        avg_accuracy = sum(accuracies.values()) / len(accuracies)
        # Only visualize if we have all the data and accuracy improved
        # ——— Prepare CSV logging ———
        log_filename = f"clients_gpaf_valid_log.csv"
        write_header = not os.path.exists(log_filename)
        with open(log_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow([
                "round","valid_acc",
                
                ])
    
        
        if avg_accuracy > self.best_avg_accuracy:
          print(f'==visualization===')
          self.best_avg_accuracy = avg_accuracy
          self.feature_visualizer.visualize_all_clients_by_class(
            features_dict=self.current_features,
            labels_dict=self.current_labels,
            accuracies=accuracies,
            epoch=server_round,
            stage="validation"
          )
          # log to CSV
          with open(log_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([server_round, accuracies])
          
          """
          noise_dim = self.latent_dim  # Noise dimension
          label_dim = self.num_classes # Label dimension
          # Generate global features with their conditioned labels
          noise = torch.randn(self.batch_size, noise_dim).to(self.device)
          labels = sample_labels(self.batch_size, self.label_probs)
          labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
          print(f" client id : hhdhd {client_id}")
          client_idy=int(client_id)
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
        """
        return avg_accuracy, {"accuracy": avg_accuracy}
    def configure_evaluate(
      self, server_round: int, parameters: Parameters, client_manager: ClientManager
) -> List[Tuple[ClientProxy, EvaluateIns]]:
      
      """Configure the next round of evaluation."""
     
      sample_size, min_num_clients = self.num_evaluate_clients(client_manager)
      clients = client_manager.sample(
        num_clients=sample_size, min_num_clients=min_num_clients
    )
      evaluate_config = {"server_round": server_round}  # Pass the round number in config
     
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
      batch_size = self.batch_size # Example batch size
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
      
      config = {
            "round": server_round,
            "generator_params": generator_params_serialized,
            #"discriminator_params": discriminator_params_serialized
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
   
    def discriminator_update_grads(
      self,
      net: torch.nn.Module,
      weights_results: NDArrays,
      gradients_aggregated: NDArrays,
      weight_decay: float,
        ) -> NDArrays:
   
      params_dict = zip(net.state_dict().keys(), weights_results)
      state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
      net.load_state_dict(state_dict, strict=True)
      optimizer = torch.optim.Adam(
        list(net.parameters()), lr=0.0002, weight_decay=weight_decay
      )
      for params, grad_ins in zip(net.parameters(), gradients_aggregated):
        params.grad = torch.tensor(grad_ins).to(params.dtype)
      optimizer.step()
      optimizer.zero_grad()
      weights_prime = [val.cpu().numpy() for _, val in net.state_dict().items()]

      return weights_prime


    def validate_clients(self, classifier_params_with_ids):
      received_ids = [cid for cid, _ in classifier_params_with_ids]
      expected_ids = set(self.client_to_domain.keys())
      if set(received_ids) != expected_ids:
        raise ValueError(f"Received client IDs {received_ids} don’t match expected {expected_ids}")
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, flwr.common.FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate results and update generator."""
        print(f'results faillure {failures}')    
        if not results:
            return None, {}

       
        # Generate z representation using the generator
        batch_size = self.batch_size # Example batch size
        noise_dim = self.latent_dim  # Noise dimension
        label_dim = self.num_classes  # Label dimension
        # Aggregate label distributions
        global_label_counts = {}
        total_samples = 0
        # Extract features and weights from clients
        features_list = []
        weights_list = []
        domain_labels = []
        domain_indices=[]
        # Aggregate label counts
        accuracy_metrics = {}
        client_features_dict = {}  # Store features per client for later gradient computation
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
              
                # Store features for later gradient computation
                client_features_dict[client_proxy.cid] = client_features
                # Add domain labels for each sample in this client's batch
                domain_labels.extend([domain_idx] * client_features.size(0))  
                domain_indices.append(domain_idx)                   
        
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
        
        # Extract num_encoder_params from the first client's metrics
        num_encoder_params = int(results[0][1].metrics["num_encoder_params"])
        num_classifier = int(results[0][1].metrics["num_classifier_params"])
        num_discriminator = int(results[0][1].metrics["num_discriminator_params"])
        encoder_params_list = []
        classifier_params_list = []
        discriminator_params_list=[]
        num_samples_list = []
        classifier_params_with_ids=[]
        
        for client_proxy, fit_res in results:
            # Convert parameters to NumPy arrays
            client_parameters = parameters_to_ndarrays(fit_res.parameters)
            client_features = pickle.loads(base64.b64decode(features_serialized.encode('utf-8')))
            client_id=client_proxy.cid
            # Validate parameter length
            if len(client_parameters) < num_encoder_params:
                print(f"Warning: Client parameters length ({len(client_parameters)}) is less than num_encoder_params ({num_encoder_params})")
                continue
            encoder_params = client_parameters[:num_encoder_params]
            classifier_params = client_parameters[num_encoder_params : num_encoder_params + num_classifier]
            discriminator_params = client_parameters[num_encoder_params + num_classifier : num_encoder_params + num_classifier + num_discriminator]
            encoder_params_list.append(encoder_params)
            classifier_params_list.append(classifier_params)
            classifier_params_with_ids.append((client_id, classifier_params))
            discriminator_params_list.append(discriminator_params)
            num_samples_list.append(fit_res.num_examples)

        # Train server discriminator
        all_features = torch.cat(features_list, dim=0)
      
        #domain_one_hot = F.one_hot(all_domain_labels, num_classes=len(results)).float()
      
        if not encoder_params_list or not classifier_params_list:
            print("No valid parameters to aggregate")
            return None, {}
        # Aggregate parameters using FedAvg
        aggregated_encoder_params = self._fedavg_parameters(encoder_params_list, num_samples_list)
        aggregated_classifier_params = self._fedavg_parameters(classifier_params_list, num_samples_list)
        aggregated_discriminator_params = self._fedavg_parameters(discriminator_params_list, num_samples_list)
        self.averagedclasssifier=aggregated_classifier_params
        #aggregate local disriminators grads
        '''
        grads_results: List[Tuple[NDArrays, int]] = [
                (fit_res.metrics["grads"], fit_res.num_examples)  # type: ignore
                for _, fit_res in results
            ]
      
        grads_results: List[Tuple[NDArrays, int]] = [
    (pickle.loads(base64.b64decode(fit_res.metrics["grads"])), fit_res.num_examples)
    for _, fit_res in results
]
        weight_decay_ = 0.0001
        
        gradients_aggregated  = aggregate(grads_results)
        weights_prime = self.discriminator_update_grads(
                self.local_discriminator,
                aggregated_discriminator_params,
                gradients_aggregated,
                weight_decay_,
            )
        disc_parameters_aggregated = weights_prime
        # Combine aggregated parameters
        '''
        aggregated_params = aggregated_encoder_params + aggregated_classifier_params + aggregated_discriminator_params 
        
        #train the globel generator
        
        self._train_generator(classifier_params_with_ids,self.label_probs,classifier_params_list , self.averagedclasssifier)
     
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
            
        }
        # Clear memory
        del features_list
        del all_features
       
        
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
        classifier = Classifier(latent_dim=self.latent_dim, num_classes=self.num_classes).to(self.device)
       
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

    def kl_divergence(self,mu, logvar):
       kl=-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
       return torch.mean(kl)


    def update_global_model(self, averaged_classifier) -> None:
      if not averaged_classifier:
        raise ValueError("classifier_params is empty")
        
      # Initialize aggregated parameters
      # Assume all clients have the same model architecture
      aggregated_params = averaged_classifier
    
      # Create or update global_model
      self.global_model = self._create_client_model(aggregated_params)
      self.global_model.to(self.device)
      self.global_model.eval()
      
    def _train_generator(self,classifier_ids,label_probs, classifier_params:  List[NDArrays] ,averaged_classifier):
     with self.mlflow.start_run(run_id=self.server_run_id):  
 
      """Train the generator using the ensemble of client classifiers."""
      noise_dim = 64
      label_dim = self.num_classes
      hidden_dim = 256
      output_dim = 64
      batch_size = self.batch_size
      learning_rate = 0.001
      num_epochs = 40
      num_domains=2
      lambda_kl = 0.7  # Tune this
      lambda_adv=0.4
      alpha = 0.5  # Dirichlet concentration parameter (tune this)
      temperature = 2.0  # Temperature for softening global model predictions
      # Optimizer
      # ensure class attributes know about it
      device=self.device 
      print(f'global server generator device{device}')
      self.generator.to(device)
      self.discriminator.to(device)
      optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
      optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate)
      criterion = nn.CrossEntropyLoss()
      bce_loss =  nn.BCEWithLogitsLoss()
      # Update global_model using FedAvg
      #self.update_global_model(averaged_classifier)
      # Mixed‑precision scalers
      scaler_d = GradScaler()
      scaler_g = GradScaler()
      for epoch in range(num_epochs):
          print('====== Training Generator=====')
          epoch_loss = 0.0
          # Sample labels from Dirichlet distribution
          # Sample labels and prepare tensors
          labels = sample_labels(batch_size, label_probs)
          labels = torch.tensor(labels, dtype=torch.long).to(self.device)
          labels_one_hot = F.one_hot(labels, num_classes=label_dim).float().to(self.device)
          # Sample noise
          noise = torch.randn(batch_size, 64, dtype=torch.float32).to(self.device)
          #noise = torch.tensor(noise, dtype=torch.float32)
          optimizer.zero_grad()
          logits = []
          # Train Discriminator with new client feedback
          optimizer_d.zero_grad()
          with autocast():
           z ,mu, logvar = self.generator(noise, labels_one_hot,return_distribution=True)
           real_z = torch.randn(batch_size, output_dim).to(self.device)
           real_labels = torch.ones(batch_size, 1).to(self.device)
           fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
           d_loss = 0.5 * (bce_loss(self.discriminator(real_z), real_labels) +
                               bce_loss(self.discriminator(z.detach()), fake_labels))
          scaler_d.scale(d_loss).backward()
          scaler_d.step(optimizer_d)
          scaler_d.update()
          with autocast():
           z ,mu, logvar = self.generator(noise, labels_one_hot,return_distribution=True)

           for params in classifier_params:
                # Convert parameters to proper format if they're named parameters
                if hasattr(params, '__iter__') and not isinstance(params, (list, tuple, np.ndarray)):
                    params = [p.detach().cpu().numpy() for _, p in params]
                
                client_model = self._create_client_model(params)
                client_model = client_model.to(self.device)
                client_model.eval()
                
                client_logits = client_model(z)
                logits.append(client_logits)
                    
           if not logits:
                raise ValueError("No valid logits generated from client models")
                
           # Average the logits from all clients
           avg_logits = torch.mean(torch.stack(logits), dim=0)
           # Compute loss
           distill_loss = criterion(avg_logits, labels)
           kl_loss = self.kl_divergence(mu, logvar)
           adv_loss = bce_loss(self.discriminator(z), real_labels)
           with torch.no_grad():
               preds = discriminator(z)  # [B, 1], after removing sigmoid
               print("global Discriminator output mean:", preds.mean().item())
           # Total loss
           total_loss = distill_loss +lambda_kl *kl_loss + lambda_adv * adv_loss
  
          scaler_g.scale(total_loss).backward()
          scaler_g.step(optimizer)
          scaler_g.update()
            
          epoch_loss += total_loss.item()
          # Log metrics using mlflow directly
          self.mlflow.log_metrics({
                    "generator_loss": epoch_loss,
                    "distill_loss": distill_loss.item(),
                    "kl_loss": kl_loss.item(),
                   "adv_loss": adv_loss.item(),
                    "epoch": epoch,
                }, step=epoch)

       
          print(f"distill loss Epoch [{epoch + 1}/{num_epochs}], distill Loss: {distill_loss:.4f}")
          print(f"kl_loss Epoch [{epoch + 1}/{num_epochs}], kl_loss Loss: {kl_loss:.4f}")
          print(f"adv_loss Epoch [{epoch + 1}/{num_epochs}], adv_loss Loss: {adv_loss:.4f}")

      save_dir = "z_representations"
    
  
