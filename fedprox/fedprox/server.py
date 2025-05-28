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
from sklearn.manifold import TSNE
from collections import defaultdict
from sklearn.metrics import pairwise_distances
from typing import List, Tuple, Optional, Dict, Callable, Union
from flwr.common.typing import NDArrays, Scalar
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import json
from flwr.server.strategy import Strategy,FedAvg
from fedprox.models import test,test_gpaf 
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

        #clusters parameters

        self.num_clusters = 4
        self.cluster_prototypes = None  # {cluster_id: {class_id: prototype}}
        self.client_assignments = {}  # {client_id: cluster_id}
      
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
        self.save_dir="visualizations"
        
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
         
        #on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        # Initialize the generator and its optimizer here
        self.num_classes =num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_avg_accuracy=0.0
        # Initialize the generator and its optimizer here
       
        self.label_probs = {label: 1.0 / self.num_classes for label in range(self.num_classes)}
        # Store client models for ensemble predictions
        self.client_classifiers = {}
        self.feature_visualizer =StructuredFeatureVisualizer(
        num_clients=2,  # total number of clients
        num_classes=self.num_classes,           # number of classes in your dataset

save_dir="feature_visualizations_gpaf"
         )
               
   
         
    def num_evaluate_clients(self, client_manager: ClientManager) -> Tuple[int, int]:
      """Return the sample size and required number of clients for evaluation."""
      num_clients = client_manager.num_available()
      return max(int(num_clients * self.fraction_evaluate), self.min_evaluate_clients), self.min_available_clients
    
    def _initialize_clusters(self, all_prototypes):
        """Initialize cluster prototypes using first num_clusters clients"""
        initial_prototypes = [all_prototypes[i] for i in range(self.num_clusters)]
        return {
            cluster_id: initial_prototypes[cluster_id]
            for cluster_id in range(self.num_clusters)
        }

    
    def _e_step(self, all_prototypes, client_ids):
        """Hard assignment of clients to clusters"""
        assignments = {}
        for client_id, prototypes in zip(client_ids, all_prototypes):
            min_dist = float('inf')
            best_cluster = 0
            
            # Calculate distance to each cluster
            for cluster_id in self.cluster_prototypes:
                total_dist = 0
                for class_id in prototypes:
                    if class_id in self.cluster_prototypes[cluster_id]:
                        # L2 distance between prototypes
                        client_proto = prototypes[class_id]
                        cluster_proto = self.cluster_prototypes[cluster_id][class_id]
                        total_dist += np.linalg.norm(client_proto - cluster_proto)
                
                if total_dist < min_dist:
                    min_dist = total_dist
                    best_cluster = cluster_id
                    
            assignments[client_id] = best_cluster
        return assignments

    
    def _m_step(self, all_prototypes, client_ids, assignments):
        """Update cluster prototypes based on assignments"""
        cluster_accumulators = defaultdict(lambda: defaultdict(list))
        
        # Accumulate prototypes per cluster and class
        for client_id, prototypes in zip(client_ids, all_prototypes):
            cluster_id = assignments[client_id]
            for class_id, proto in prototypes.items():
                cluster_accumulators[cluster_id][class_id].append(proto)
                
        # Compute mean prototypes
        new_clusters = defaultdict(dict)
        for cluster_id in cluster_accumulators:
            for class_id in cluster_accumulators[cluster_id]:
                protos = cluster_accumulators[cluster_id][class_id]
                new_clusters[cluster_id][class_id] = np.mean(protos, axis=0)
                
        # Handle empty clusters (re-initialize with random client)
        for cluster_id in range(self.num_clusters):
            if cluster_id not in new_clusters:
                random_client = np.random.choice(client_ids)
                new_clusters[cluster_id] = all_prototypes[client_ids.index(random_client)]
                
        return new_clusters
    
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

        # Prepare config for next round
        config = {
            "server_round": server_round,
            
        }
        clients_params_list=[]
        num_samples_list=[]
        self.client_prototypes = {}  # <-- ADD THIS LINE
        for client_proxy, fit_res in results:
                client_id=client_proxy.cid
                #prototypes = fit_res.metrics.get("prototypes").encode('utf-8')
                #prototypes = pickle.loads(base64.b64decode(prototypes))
                client_parameters = parameters_to_ndarrays(fit_res.parameters)
                clients_params_list.append(client_parameters)
                """
                if prototypes:
                    self.client_prototypes[client_id] = prototypes
                """
                num_samples_list.append(fit_res.num_examples)
        # Cluster clients using cosine similarity between prototype vectors
        #self.perform_clustering(server_round)
        #aggregated_params = super().aggregate_fit(server_round, client_parameters, failures)
        aggregated_params = self._fedavg_parameters(clients_params_list, num_samples_list)
        #*** compute the clustering algorithm ***#
        client_ids = [client.cid for client,_ in results]
        #print(f' client ids {client_ids}')

        
        prototypes = [pickle.loads(base64.b64decode(r.metrics["prototypes"])) for _,r in results]
        #print(f'prototypes: **** {prototypes} ****')
        # Convert prototypes to numpy arrays
        proto_arrays = []
        for p in prototypes:
            proto_arrays.append({
                cls: np.array(proto) 
                for cls, proto in p.items()
            })
        
        # Initialize clusters if first round
        if self.cluster_prototypes is None:
            self.cluster_prototypes = self._initialize_clusters(proto_arrays)
            
        # 4. EM Algorithm
        # E-step: Assign clients to clusters
        assignments = self._e_step(proto_arrays, client_ids)
        
        # M-step: Update cluster prototypes
        self.cluster_prototypes = self._m_step(proto_arrays, client_ids, assignments)
        
        # 5. Update client assignments
        self.client_assignments.update(assignments)
        
        # 6. Prepare cluster prototypes for next round
        for cluster_id in self.cluster_prototypes:
            for class_id in self.cluster_prototypes[cluster_id]:
                if isinstance(self.cluster_prototypes[cluster_id][class_id], np.ndarray):
                    self.cluster_prototypes[cluster_id][class_id] = \
                        self.cluster_prototypes[cluster_id][class_id].tolist()

        self._visualize_clusters(prototypes, client_ids, server_round)
        return ndarrays_to_parameters(aggregated_params),config
    

    def _visualize_clusters(self, prototypes, client_ids, server_round):
      # Flatten prototypes (average across classes)
      prototype_matrix = []
      for client_prototypes in prototypes:
        client_proto = np.mean(list(client_prototypes.values()), axis=0)
        prototype_matrix.append(client_proto)
      prototype_matrix = np.array(prototype_matrix)

      # Project with t-SNE
      n_clients= len(prototype_matrix)
      perplexity = min(30, max(1, n_clients - 1))  # Ensures 1 <= perplexity < n_clients

      tsne = TSNE(n_components=2,  perplexity=perplexity,random_state=42 )
      projections = tsne.fit_transform(prototype_matrix)

      # Get cluster assignments
      cluster_assignments = [self.client_assignments.get(cid, -1) for cid in client_ids]  # -1 = unassigned
      print(f" cluster assignment {cluster_assignments}")
      # Create plot
      plt.figure(figsize=(12, 8))
      scatter = plt.scatter(
        projections[:, 0], projections[:, 1], 
        c=cluster_assignments, cmap='tab10', alpha=0.7
     )
    
      # Annotate points with client IDs
      for i, (x, y) in enumerate(projections):
        plt.text(x, y, client_ids[i], fontsize=8, ha='center', va='bottom')
    
      # Add legend and labels
      plt.legend(*scatter.legend_elements(), title="Cluster ID")
      plt.title(f"Client Prototypes (Round {server_round})\nColors = Cluster ID, Labels = Client ID")
      plt.xlabel("t-SNE 1")
      plt.ylabel("t-SNE 2")
    
      # Save and display
      plt.savefig(f"clusters_round_{server_round}.png", dpi=300, bbox_inches='tight')
      plt.show()
      plt.close()

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
        # ——— Prepare CSV logging ———
        log_filename = f"clients_acc_valid_{client_id}_log.csv"
        write_header = not os.path.exists(log_filename)
        with open(log_filename, 'a', newline='') as csvfile:
          writer = csv.writer(csvfile)
          if write_header:
            writer.writerow([
                "round","accuracy_avg",
               
                ])
        
        if avg_accuracy > self.best_avg_accuracy:

          print(f'==visualization===')
          # log to CSV
          with open(log_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([server_round, avg_accuracy])
  
    
          '''
          self.best_avg_accuracy = avg_accuracy
          self.feature_visualizer.visualize_all_clients_by_class(
            features_dict=self.current_features,
            labels_dict=self.current_labels,
            accuracies=accuracies,
            epoch=server_round,
            stage="validation"
          )
          '''
         
         
        return avg_accuracy, {"accuracy": avg_accuracy}
   
    def configure_fit(self, server_round, parameters, client_manager):
      # Sample clients
      num_clients_per_round = int(self.fraction_fit * client_manager.num_available())
      selected_clients = client_manager.sample(
        num_clients=num_clients_per_round,
        min_num_clients=4,
      )
    
      # ROUND 1: No clusters yet
      if server_round == 1:
        for client in selected_clients:
            # Send empty config (no regularization)
            client.config.update({
                "global_prototypes": {},
                "N_j": {}
            })
        return selected_clients
    
      # ROUND 2+: Use clusters
      for client in selected_clients:
        # Handle unassigned clients (new or missed)
        if client.cid not in self.client_assignments:
            # Assign to random cluster or default cluster
            cluster_id = np.random.randint(0, self.num_clusters)
            self.client_assignments[client.cid] = cluster_id
        
        cluster_id = self.client_assignments[client.cid]
        client_prototypes = self.cluster_prototypes[cluster_id]
        N_j = {cls: self.cluster_class_counts[cluster_id][cls] 
               for cls in client_prototypes}
        
        client.config.update({
            "global_prototypes": client_prototypes,
            "N_j": N_j
        })
    
      return selected_clients
        
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
      

  

