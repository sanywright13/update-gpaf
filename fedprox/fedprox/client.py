"""Defines the MNIST Flower Client and a function to instantiate it."""

from collections import OrderedDict
from typing import Callable, Dict, List, Tuple
from flwr.common import Context
import flwr as fl
import numpy as np
import torch
import copy
import csv
from collections import defaultdict

import torch.nn.functional as F
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import json
from flwr.client import NumPyClient, Client
from flwr.common import ConfigsRecord, MetricsRecord, ParametersRecord ,Context, ConfigRecord
from  mlflow.tracking import MlflowClient
import base64
import pickle
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
import os
from fedprox.models import train_gpaf,test_gpaf,init_net,train_moon,test_moon,save_client_model,load_client_model,Model
from fedprox.dataset_preparation import compute_label_counts, compute_label_distribution
from fedprox.features_visualization import extract_features_and_labels,StructuredFeatureVisualizer
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, net, 
     data,validset,
     local_epochs,
     client_id,
      mlflow,
      run_id,
      feature_visualizer
      ,
            device,batch_size):
        self.net = net
        
        self.traindata = data
        self.validdata=validset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_epochs=local_epochs
        self.client_id=client_id
        self.num_classes=9
        self.num_clients=2
        self.batch_size=batch_size
        
        print(f"dd Batch size client side : {self.batch_size}")
        # Move models to device
        self.net.to(self.device)
       
        self.domain_dim=32
       
        self. mlflow= mlflow
    
        self.run_id=run_id
        self.feature_visualizer=feature_visualizer
        # Initialize dictionaries to store features and labels
        self.client_features = {}  # Add this
        self.client_labels = {}    # Add this
       
    #update the local model with parameters received from the server
    def set_parameters(self, parameters: List[np.ndarray]):
      params_dict = zip(self.net.state_dict().keys(), parameters)
      state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
      self.net.load_state_dict(state_dict, strict=True)

    #get the updated model parameters from the local model return local model parameters
    
    def get_parameters(self , config: Dict[str, Scalar] = None):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]


   
    #second and call set_para  
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        print(f'===evaluate client=== {type(parameters)}')
        self.set_parameters(parameters)

        for batch_idx, (data, target) in enumerate(self.validdata):
          print(f"evaluate dd Batch {batch_idx}, data shape: {data.shape}, target shape: {target.shape}")
          break  # J
        loss, accuracy = test_gpaf(self.net, self.validdata, self.device)

        # Extract features and labels
        val_features, val_labels = extract_features_and_labels(
        self.net,
        self.validdata,
        self.device
           )
    
        if val_features is not None:
          self.client_features[self.client_id] = val_features
          self.client_labels[self.client_id] = val_labels

        with self.mlflow.start_run(run_id=self.run_id):  
            print(f' config client {config.get("server_round")}')
            self.mlflow.log_metrics({
                f"client_{self.client_id}/eval_loss": float(loss),
                f"client_{self.client_id}/eval_accuracy": float(accuracy),
                #f"client_round":float(round_number),
               # f"client_{self.client_id}/eval_samples": samples
            }, step=config.get("server_round"))
            # Also log in format for easier plotting
        
        #visualize all clients features per class
        features_np = val_features.detach().cpu().numpy()
        labels_np = val_labels.detach().cpu().numpy().reshape(-1)  # Ensure 1D array
        # In client:
        features_serialized = base64.b64encode(pickle.dumps(features_np)).decode('utf-8')
        labels_serialized = base64.b64encode(pickle.dumps(labels_np)).decode('utf-8')
        print(f"Client {self.client_id} sending features shape: {features_np.shape}")
        print(f"Client {self.client_id} sending labels shape: {labels_np.shape}")
         
        print(f'client id : {self.client_id} and valid accuracy is {accuracy} and valid loss is : {loss}')
        return float(loss), len(self.validdata), {"accuracy": float(accuracy),
         "features": features_serialized,
            "labels": labels_serialized,
        }
    
    
    
   
    
    def fit(self, parameters, config):
        """Train local models using latest generator state."""
        #print(f'=== client training {config}')
        # Update local models with global parameters
        self.set_parameters(parameters)
        
        all_labels = torch.cat(all_labels).squeeze().to(self.device)
        label_distribution = compute_label_distribution(all_labels, self.num_classes)
        # Serialize the label distribution to a JSON string
        label_distribution_str = json.dumps(label_distribution)
       
        train_gpaf(self.net, self.traindata,self.device,self.client_id,self.local_epochs,self.batch_size)
           
        # Extract features for server
        """
        features = []
        with torch.no_grad():
          for data, labels in self.traindata:

            data = data.to(self.device)
            if labels.dim() > 1:
                labels = labels.squeeze()
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)  # Handle single sample
            labels_onehot = F.one_hot(labels.long(), num_classes=self.num_classes).float()

            feat = self.encoder(data)
            features.append(feat.cpu().numpy())
    
        # Concatenate all features
        all_features = np.concatenate(features, axis=0)
        all_features_serialized = base64.b64encode(pickle.dumps(all_features)).decode('utf-8')
        
        # Clear memory
        del features
        del all_features
        """

        #protoype

        # === Prototype Extraction ===
        self.net.eval()
        class_embeddings = defaultdict(list)

        with torch.no_grad():
          for batch in self.trainloader:
            images, labels = batch
            images, labels = images.to(DEVICE, dtype=torch.float32), labels.to(DEVICE, dtype=torch.long)
            h, _, _ = self.net(images)  # Get encoder output (before projection head)

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_embeddings[label].append(h[i].cpu())

        # Compute prototypes: mean of embeddings per class
        prototypes = {}
        for class_id in range(self.num_classes):
          if class_id in class_embeddings and len(class_embeddings[class_id]) > 0:
            prototypes[class_id] = torch.stack(class_embeddings[class_id]).mean(dim=0)
          else:
            prototypes[class_id] = torch.zeros_like(h[0].cpu())

        return (
        self.get_parameters(),
        len(self.traindata),
        {
           
            "label_distribution": label_distribution_str,
            "prototypes": prototypes,
            #"grads": grads_serialized

        
        }
    )



def gen_client_fn(
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    learning_rate: float,
    model=None,
experiment_name =None,
strategy='fedavg',
cfg=None  ,
 device=torch.device,batch_size=13

) -> Callable[[Context], Client]:  # pylint: disable=too-many-arguments
    import mlflow
    # be a straggler. This is done so at each round the proportion of straggling
    client = MlflowClient()
    print(f"1 : client Using device: {device}")  # Ensure it's either CUDA or CPU
    def client_fn(context: Context) -> Client:
        # Access the client ID (cid) from the context
      cid = context.node_config["partition-id"]
      # Create or get experiment
      experiment = mlflow.get_experiment_by_name(experiment_name)
      if "mlflow_id" not in context.state.config_records:
            context.state.config_records["mlflow_id"] = ConfigRecord()
          
      print(f"2fff : client Using device: {device}")  # Ensure it's either CUDA or CPU

      run_ids = context.state.config_records["mlflow_id"]

      if str(cid) not in run_ids:
            run = client.create_run(experiment.experiment_id)
            run_ids[str(cid)] = [run.info.run_id]
    
      with mlflow.start_run(experiment_id=experiment.experiment_id, run_id=run_ids[str(cid)][0],nested=True) as run:
        run_id = run.info.run_id
        print(f"Created MLflow run for client {cid}: {run_id}")
        
        input_dim = 28  # Example: 28x28 images flattened
        hidden_dim = 128
        latent_dim = 128
        num_classes = 9
        num_epochs=3
        
               
        if strategy=="gpaf":
          
          img_shape=(28,28)
          net = Model(out_dim=256, n_classes=9)
     
          trainloader = trainloaders[int(cid)]
          # Initialize the feature visualizer for all clients
          feature_visualizer = StructuredFeatureVisualizer(
        num_clients=num_clients,  # total number of clients
num_classes=num_classes,
save_dir="feature_visualizations"
          )
          #print(f'  ffghf {trainloader}')
          valloader = valloaders[int(cid)]
          for batch_idx, (data, target) in enumerate(valloader):
            print(f"Batch {batch_idx}, data shape: {data.shape}, target shape: {target.shape}")
            break  # Just check the first batch
          numpy_client =  FederatedClient(
            net,
           
            trainloader,
            valloader,
            num_epochs,
            cid,
            mlflow
            ,
            run_id,
            feature_visualizer,
            device,batch_size

          )

         
        elif strategy =="moon":
          
          trainloader = trainloaders[int(cid)]
          testloader = valloaders[int(cid)]
          return MOONFlowerClient(
            int(cid),
            cfg.output_dim,
            trainloader,
            testloader,
            device,
            num_epochs,        
            cfg.mu,
            cfg.temperature,
                
          )

        else:
          # Load model
          trainloader = trainloaders[int(cid)]
          valloader = valloaders[int(cid)]
          latent_dim=128
          num_classes=9
          model = EncoderClassifier(latent_dim=latent_dim, num_classes=num_classes)
          numpy_client = FlowerClient(
            model, trainloader, valloader,num_epochs,
           cid,run_id,mlflow)

        return numpy_client.to_client()
      
    return client_fn


# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
backend_config = {"client_resources": {"num_cpus":1 , "num_gpus": 0.0}}
# When running on GPU, assign an entire GPU for each client
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
    # Refer to our Flower framework documentation for more details about Flower simulations
    # and how to set up the `backend_config`
class FlowerClient(NumPyClient):

    def __init__(self, net, trainloader, valloader,local_epochs,partition_id,run_id,mlflow):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs=local_epochs
        self.client_id=partition_id
        self.run_id=run_id
        self.mlflow=mlflow
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #update the local model with parameters received from the server
    def set_parameters(self,net, parameters: List[np.ndarray]):
      params_dict = zip(net.state_dict().keys(), parameters)
      state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
      net.load_state_dict(state_dict, strict=True)

    #get the updated model parameters from the local model return local model parameters
    
    def get_parameters(self , config: Dict[str, Scalar] = None):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    #get parameters from server train with local data end return the updated local parameter to the server
    def fit(self, parameters, config):

        self.set_parameters(self.net, parameters)
        self.train(self.net, self.trainloader,self.client_id,epochs=self.local_epochs)
        # Log the model after training
        """
        with mlflow.start_run(run_id=self.run_ids[str(self.client_id)][0], nested=True) as run:
            mlflow.pytorch.log_model(self.net, f"model_client_{self.client_id}")
        """
        return self.get_parameters(self.net), len(self.trainloader), {}


    
    def evaluate(self, parameters, config):
       
          server_round = config["server_round"]
          #print(f"Client {self.client_id} round id after training: {server_round}")
          self.set_parameters(self.net, parameters)
          loss, accuracy = self.test(self.net, self.valloader)
          print(f"Client {self.client_id} round id {server_round} , val accuracy: {accuracy}")
          #print(f'****evaluation**** {mlflow}')
          with self.mlflow.start_run(run_id=self.run_id):  
            self.mlflow.log_metrics({
                f"client_{self.client_id}/eval_loss": float(loss),
                f"client_{self.client_id}/eval_accuracy": float(accuracy),
               
            }, step=config.get("server_round"))
            # Also log in format for easier plotting
          print(f'client id : {self.client_id} and valid accuracy is {accuracy} and valid loss is : {loss}')
          # Extract features and labels
          val_features, val_labels = extract_features_and_labels(
          self.net,
         self.valloader,
          self.device
           )
          #visualize all clients features per class
          features_np = val_features.detach().cpu().numpy()
          labels_np = val_labels.detach().cpu().numpy().reshape(-1)  # Ensure 1D array
          # In client:
          features_serialized = base64.b64encode(pickle.dumps(features_np)).decode('utf-8')
          labels_serialized = base64.b64encode(pickle.dumps(labels_np)).decode('utf-8')
         
          return float(loss), len(self.valloader), {"accuracy": float(accuracy),
         "features": features_serialized,
            "labels": labels_serialized,
          }
    
    
    def test(self,net, testloader):
      """Evaluate the network on the entire test set."""
      DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      net.to(DEVICE)
      criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
      correct, total, loss = 0, 0, 0.0
      num_classes=9
      net.eval()
      # Initialize metrics
     
      print(f' ==== client test func')
      with torch.no_grad():
        for batch in testloader:
            images, labels = batch
            images, labels = images.to(DEVICE ,  non_blocking=True), labels.to(DEVICE  ,  non_blocking=True)

            #labels=labels.squeeze(1)
            
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            
            
      
      loss /= len(testloader.dataset)
      accuracy = correct / total
      print(f"Test Accuracy: {accuracy:.4f}")
     
      return loss, accuracy   

#monn client side

class MOONFlowerClient(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        # net: torch.nn.Module,
        net_id: int,
       
        output_dim: int,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        mu: float,
        temperature: float,
    ):  # pylint: disable=too-many-arguments
 
        self.net = init_net(output_dim)
        self.net_id = net_id
        #self.dataset = dataset
        
        self.output_dim = output_dim
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = 0.00013914064388085564
        self.mu = mu  # pylint: disable=invalid-name
        self.temperature = temperature
        self.model_dir="moon"
        self.client_id=net_id
        #self.model_dir = model_dir
        #self.alg = alg
        self.global_net=init_net(output_dim)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Change the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        self.set_parameters(parameters)
        print(f'model output :{self.output_dim}')
        prev_net = init_net(self.output_dim)
      
        if not os.path.exists(os.path.join(self.model_dir, str(self.net_id))):
            prev_net = copy.deepcopy(self.net)
        else:
            # load previous model from model_dir
            prev_net.load_state_dict(
                torch.load(
                    os.path.join(self.model_dir, str(self.net_id), "prev_net.pt")
                )
            )
        global_net = init_net(self.output_dim)
        global_net.load_state_dict(self.net.state_dict())
        self.global_net=global_net
        train_moon(
                self.net,
                global_net,
                prev_net,
                self.trainloader,
                self.num_epochs,
                self.learning_rate,
                self.mu,
                self.temperature,
                self.device,
                self.client_id
                )
        
        
        if not os.path.exists(os.path.join(self.model_dir, str(self.net_id))):
            os.makedirs(os.path.join(self.model_dir, str(self.net_id)))
        torch.save(
            self.net.state_dict(),
            os.path.join(self.model_dir, str(self.net_id), "prev_net.pt"),
        )
     
        return self.get_parameters({}), len(self.trainloader), {"is_straggler": False}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        self.set_parameters(parameters)
        # skip evaluation in the client-side
        loss = 0.0
        accuracy = 0.0
        print(f'client id : {self.client_id} and valid accuracy is {accuracy} and valid loss is : {loss}')
        # Extract local features and labels
        val_features, val_labels = extract_features_and_labels(
          self.net,
         self.valloader,
          self.device
           )

        
        # Extract global features and labels
        if self.global_net is not None:
          print("========global features exist==========")
          global_features,global_labels = extract_features_and_labels(
          self.global_net,
         self.valloader,
          self.device
           )
          global_features_np = global_features.detach().cpu().numpy()
          global_features_serialized = base64.b64encode(pickle.dumps(global_features_np)).decode('utf-8')
          global_labels_np = global_labels.detach().cpu().numpy().reshape(-1)  # Ensure 1D array
          global_labels_serialized = base64.b64encode(pickle.dumps(global_labels_np)).decode('utf-8')

        else:
          global_features_serialized=""
        accuracy , loss = test_moon(self.net, self.valloader, device="cpu")

        #visualize all clients features per class
        features_np = val_features.detach().cpu().numpy()
        labels_np = val_labels.detach().cpu().numpy().reshape(-1)  # Ensure 1D array
        # In client:
        features_serialized = base64.b64encode(pickle.dumps(features_np)).decode('utf-8')
        labels_serialized = base64.b64encode(pickle.dumps(labels_np)).decode('utf-8')
        
        

        
        
        return float(loss), len(self.valloader), {"accuracy": float(accuracy),
         "features": features_serialized,
         "global_features": global_features_serialized,
            "labels": labels_serialized,
            "global_labels":global_labels_serialized
        }
       


  # Save the trained model to MLflow.    

  # Save the trained model to MLflow.    
