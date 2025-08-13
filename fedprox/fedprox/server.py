from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple
#from MulticoreTSNE import print_function
import flwr
import mlflow
from torch.cuda.amp import autocast, GradScaler
import base64
import pickle
import datetime
from numpy.linalg import norm
from matplotlib import cm
from matplotlib.colors import ListedColormap
from torch.distributions import Dirichlet, Categorical
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from matplotlib import cm
import random
from flwr.common import GetPropertiesIns
import json
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
from fedprox.client_monitoring import CRACS_MDA, load_log_data
from flwr.server.strategy import Strategy,FedAvg
from fedprox.models import test,test_gpaf 
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.client_proxy import ClientProxy
from fedprox.features_visualization import extract_features_and_labels,StructuredFeatureVisualizer
import csv
import requests
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
import os
from fedprox.client_monitoring import update_histories
from fedprox.client_monitoring import T_hat

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
        self.server_url = "https://add18b7094f7.ngrok-free.app/heartbeat"
        self.fairness_k=2
        #clusters parameters

        self.num_clusters = 4
        self.client_assignments = {}  # {client_id: cluster_id}
        self.global_T_max = 0.0  # <--- THIS IS THE FIX
        self.trained_clients = set()
        self.client_prototype_status = {}

        # Initialize as empty dictionaries
        self.cluster_prototypes = {i: {} for i in range(self.num_clusters)}
        self.cluster_class_counts = {i: defaultdict(int) for i in range(self.num_clusters)}
        
        
        # CSMDA Client Selection Parameters (UPDATED)
        self.training_times = defaultdict(float)
        self.selection_counts = defaultdict(int)
        self.accuracy_history = defaultdict(float)
        self._current_accuracies = {}
        # In server class
        self.client_prototype_history = defaultdict(dict)  # {client_id: {round: prototypes}}
      


        # NEW/MODIFIED FAIRNESS ATTRIBUTES
        initial_target_selections= 3
        max_target_selections = 10
        reliability_lambda = 0.05
        acc_drop_threshold  = 0.005
        self.client_targets = defaultdict(lambda: initial_target_selections)
        self.initial_target_selections = initial_target_selections
        self.max_target_selections = max_target_selections
        self.acc_drop_threshold = acc_drop_threshold

        # NEW RELIABILITY ATTRIBUTE
        self.reliability_lambda = reliability_lambda

        self.phase_threshold = 30
        
        # CSMDA Hyperparameters
        self.alpha = 0.3  # EMA decay for training time
        self.beta = 0.2   # fairness boost increment
        self.epsilon = 0.1  # straggler tolerance (10% of T_max)
        self.target_selections = 5  # minimum selections per client
        self.accuracy_eval_interval = 5  # evaluate accuracy every R rounds
        self.phase_threshold = 30  # switch from reliability to fairness focus
        

        # Initialize other components
        self.stat_util = {}
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_avg_accuracy = 0.0
        self.batch_size = batch_size
        self.save_dir = "visualizations"

        # === FIX: Add this line to initialize the cache ===
        self.client_prototypes_cache = {} 

        # --- REVISED: Straggler Simulation Setup ---
        # Initialize an empty dictionary. It will be populated later.
        self.client_straggler_profiles = {}
        # Store the percentages, so we can use them later
        self.permanent_straggler_percentage = 0.1
        self.occasional_straggler_percentage = 0.2


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
    def _setup_straggler_profiles(self, client_manager):
        """
        Assigns a straggler profile to each client based on the CIDs
        provided by the ClientManager. This is a one-time operation.
        """
        if self.client_straggler_profiles:
            # Profiles are already set, so do nothing.
            return

        all_client_cids = list(client_manager.all().keys())
        num_clients = len(all_client_cids)
        
        num_permanent_stragglers = int(num_clients * self.permanent_straggler_percentage)
        num_occasional_stragglers = int(num_clients * self.occasional_straggler_percentage)

        shuffled_ids = random.sample(all_client_cids, num_clients)
        
        for i, cid in enumerate(shuffled_ids):
            if i < num_permanent_stragglers:
                self.client_straggler_profiles[cid] = "permanent"
            elif i < num_permanent_stragglers + num_occasional_stragglers:
                self.client_straggler_profiles[cid] = "occasional"
            else:
                self.client_straggler_profiles[cid] = "normal"
        
        print(f"Straggler Profiles Initialized: {num_permanent_stragglers} permanent, {num_occasional_stragglers} occasional.")

    def num_evaluate_clients(self, client_manager: ClientManager) -> Tuple[int, int]:
      """Return the sample size and required number of clients for evaluation."""
      num_clients = client_manager.num_available()
      return max(int(num_clients * self.fraction_evaluate), self.min_evaluate_clients), self.min_available_clients
   
 
    # --- MODIFIED aggregate_fit: REMOVED CLUSTERING LOGIC ---
    def aggregate_fit(
      self,
    server_round: int,
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
      print(f'results failure: {failures}')
      if not results:
        print("No clients returned results. Aggregation skipped.")
        return None, {}

      clients_params_list = []
      num_samples_list = []
  
      # To compute the global T_max, we'll need a list of durations from this round
      current_round_durations = []

      for client_proxy, fit_res in results:
        client_id = client_proxy.cid
        metrics = fit_res.metrics
        
        # 1. Update EWMA for client-specific training time (T_c)
        if "duration" in metrics:
            duration = metrics["duration"]
            ema_alpha = 0.3
            if client_id not in self.training_times:
                # Initialize T_c for new clients
                self.training_times[client_id] = duration
            else:
                # Update T_c using EWMA
                self.training_times[client_id] = (
                    ema_alpha * duration +
                    (1 - ema_alpha) * self.training_times[client_id]
                )
            # Add duration to the list for global T_max calculation
            current_round_durations.append(duration)
        
        if "loss_sq_mean" in metrics and "data_size" in metrics:
            stat_util = metrics["data_size"] * metrics["loss_sq_mean"]
            self.stat_util[client_id] = stat_util
        
        # 2. Extract model parameters and number of samples
        clients_params_list.append(parameters_to_ndarrays(fit_res.parameters))
        num_samples_list.append(fit_res.num_examples)

      # 3. Update the global T_max using EWMA after processing all clients
      # This ensures a stable, long-term average
      if current_round_durations:
        current_avg_duration = sum(current_round_durations) / len(current_round_durations)
        ewma_decay = 0.1 # A smaller decay for a more stable global average
        if self.global_T_max == 0.0:
            self.global_T_max = current_avg_duration
        else:
            self.global_T_max = (1 - ewma_decay) * self.global_T_max + ewma_decay * current_avg_duration

      # 4. Perform parameter aggregation (FedAvg)
      aggregated_params = self._fedavg_parameters(clients_params_list, num_samples_list)
  
      # 5. Return aggregated parameters and metrics
      return ndarrays_to_parameters(aggregated_params), {}



    def _visualize_clusters(self, prototypes, client_ids, server_round, true_domain_map=None):
      # 1. Flatten prototypes: one vector per client
      prototype_matrix = []
      for client_prototypes in prototypes:
        client_proto = np.mean(list(client_prototypes.values()), axis=0)
        prototype_matrix.append(client_proto)
      prototype_matrix = np.array(prototype_matrix)

      # 2. t-SNE projection
      n_clients = len(prototype_matrix)
      perplexity = min(30, max(1, n_clients - 1))
      tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
      projections = tsne.fit_transform(prototype_matrix)

      # 3. Cluster assignments (predicted by your method)
      cluster_assignments = [self.client_assignments.get(cid, -1) for cid in client_ids]
      unique_clusters = sorted(set(cluster_assignments))
      num_clusters = len(unique_clusters)

      # 4. Color map setup for clusters
      base_cmap = cm.get_cmap("tab20", num_clusters)
      colors = [base_cmap(i) for i in range(num_clusters)]
      color_map = ListedColormap(colors)
      cluster_id_to_color_index = {cluster_id: idx for idx, cluster_id in enumerate(unique_clusters)}
      color_indices = [cluster_id_to_color_index[cid] for cid in cluster_assignments]

      # 5. Marker setup for true domains
      markers = ['o', 's', '^', 'D', 'P', 'X']
      domain_to_marker = {}
      if true_domain_map:
        unique_domains = sorted(set(true_domain_map.get(cid, "unknown") for cid in client_ids))
        domain_to_marker = {dom: markers[i % len(markers)] for i, dom in enumerate(unique_domains)}

      # 6. Begin plotting
      plt.figure(figsize=(12, 8))

      for i, (x, y) in enumerate(projections):
        client_id = client_ids[i]
        cluster_id = cluster_assignments[i]
        color_index = cluster_id_to_color_index[cluster_id]

        if true_domain_map:
            domain = true_domain_map.get(client_id, "unknown")
            marker = domain_to_marker.get(domain, 'o')
        else:
            domain = "unknown"
            marker = 'o'

        plt.scatter(
            x, y,
            c=[colors[color_index]],
            marker=marker,
            edgecolor='k',
            s=100,
            alpha=0.8
        )
        plt.text(x, y, str(client_id), fontsize=7, ha='center', va='bottom')

      # 7. Legends
      # Cluster legend (colors)
      cluster_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {cid}',
                   markerfacecolor=colors[idx], markersize=8)
        for cid, idx in cluster_id_to_color_index.items()
    ]

      # Domain legend (markers)
      domain_handles = []
      if true_domain_map:
        for dom, marker in domain_to_marker.items():
            domain_handles.append(
                plt.Line2D([0], [0], marker=marker, color='k', label=f'Domain: {dom}',
                           markerfacecolor='gray', markersize=8, linestyle='None')
            )

      plt.legend(handles=cluster_handles + domain_handles, title="Cluster / Domain", bbox_to_anchor=(1.05, 1), loc='upper left')

      # 8. Plot aesthetics
      plt.title(f"Client Prototypes (Round {server_round})\nColors = Cluster ID, Shapes = True Domain, Labels = Client ID")
      plt.xlabel("t-SNE 1")
      plt.ylabel("t-SNE 2")
      plt.tight_layout()
      plt.savefig(f"clusters_round_{server_round}.png", dpi=300, bbox_inches='tight')
      plt.show()
      plt.close()

      # 9. Optional: Clustering quality metrics
      if true_domain_map:
        predicted_clusters = cluster_assignments
        true_domains = [true_domain_map.get(cid, -1) for cid in client_ids]

        ari = adjusted_rand_score(true_domains, predicted_clusters)
        nmi = normalized_mutual_info_score(true_domains, predicted_clusters)

        print(f"Clustering Quality: ARI = {ari:.3f}, NMI = {nmi:.3f}")


   
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
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.EvaluateRes]],
        failures: List[Union[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes], Exception]],
    ) -> Tuple[Optional[flwr.common.Parameters], Dict[str, flwr.common.Scalar]]:
        print(f"[Server] Round {server_round}: {len(results)} clients evaluated, {len(failures)} failed evaluation.")
        
        aggregated_accuracy = 0.0
        if results:
            self._current_accuracies = {}
            for client_proxy, res in results:
                client_id = client_proxy.cid
                if "accuracy" in res.metrics:
                    client_accuracy = float(res.metrics["accuracy"])
                    self._current_accuracies[client_id] = client_accuracy
                    
                    with self.mlflow.start_run(run_id=self.server_run_id):
                        self.mlflow.log_metrics({
                            f"accuracy_client_{client_id}": client_accuracy
                        }, step=server_round)
                else:
                    print(f"[Warning] Client {client_id} did not report 'accuracy' in eval_res.metrics.")

                if "features" in res.metrics and "labels" in res.metrics:
                    try:
                        features_np = pickle.loads(base64.b64decode(res.metrics.get("features").encode('utf-8')))
                        labels_np = pickle.loads(base64.b64decode(res.metrics.get("labels").encode('utf-8')))
                        pass
                    except Exception as e:
                        print(f"[Warning] Failed to decode features/labels for client {client_id}: {e}")

            if self._current_accuracies:
                aggregated_accuracy = sum(self._current_accuracies.values()) / len(self._current_accuracies)
                print(f"[Server] Round {server_round}: Aggregated Average Accuracy: {aggregated_accuracy:.4f}")
                with self.mlflow.start_run(run_id=self.server_run_id):
                    self.mlflow.log_metrics({"avg_accuracy_global": aggregated_accuracy}, step=server_round)

            if aggregated_accuracy > self.best_avg_accuracy:
                self.best_avg_accuracy = aggregated_accuracy
            
            log_filename = "server_accuracy_log.csv"
            write_header = not os.path.exists(log_filename)
            with open(log_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if write_header:
                    writer.writerow(["round", "avg_accuracy"])
                writer.writerow([server_round, aggregated_accuracy])

        else:
            print(f"[Server] Round {server_round}: No evaluation results received.")
            aggregated_accuracy = 0.0
            
        return None, {"accuracy": aggregated_accuracy}
   
    def _load_client_logs(self, server_round):
        """Load client training logs from heartbeat server"""
        try:
            # Load training time data from your heartbeat logs
            log_file = f"client_logs_round_{server_round-1}.json"
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
                return logs
            return {}
        except Exception as e:
            print(f"[Warning] Could not load client logs: {e}")
            return {}
   
    def _update_training_times(self, fit_results: List[Tuple[ClientProxy, FitRes]]):
        """Update the training times and the global T_max using EWMA."""
        for client_proxy, fit_res in fit_results:
            duration = fit_res.metrics.get("duration", 0.0)
            client_id = str(client_proxy.cid)
            
            # Store the latest training time
            self.training_times[client_id] = duration
            
            # Update the global T_max using EWMA
            if self.global_T_max == 0.0:
                self.global_T_max = duration
            else:
                self.global_T_max = (1 - self.ewma_decay) * self.global_T_max + self.ewma_decay * duration

    def _compute_reliability_scores(self, client_ids: List[str]) -> Dict[str, float]:
        """Computes reliability scores based on training duration relative to a stable T_max."""
        reliability_scores = {}
        
        # Use the global, stable EWMA T_max
        T_max = self.global_T_max if self.global_T_max > 0 else 1.0

        for client_id in client_ids:
            T_c = self.training_times.get(client_id, T_max) # Default to T_max for new clients
            penalty_term = max(0.0, T_c - T_max)
            score = np.exp(-self.reliability_lambda * penalty_term)
            
            reliability_scores[client_id] = float(score)
            
        return reliability_scores

    def _compute_fairness_scores(self, client_ids: List[str], server_round: int) -> Dict[str, float]:
        """Computes fairness scores using the new sigmoid-based formulation."""
        fairness_scores = {}
        T_total = server_round # Use current round number as T_total
        n = len(self.selection_counts) # Total number of clients who have participated
        
        if n == 0:
            return {cid: 0.5 for cid in client_ids} # Neutral score if no history
            
        for client_id in client_ids:
            v_c = self.selection_counts.get(client_id, 0)
            
            # Use the sigmoid-based score
            ideal_selections = T_total / n
            R_c = v_c / ideal_selections if ideal_selections > 0 else 0
            
            score = 1 / (1 + np.exp(self.fairness_k * (R_c - 1)))
            
            fairness_scores[client_id] = float(score)
            
        return fairness_scores
    
    def _compute_global_selection_scores(self, client_ids: List[str], server_round: int) -> Dict[str, float]:
        reliability_scores = self._compute_reliability_scores(client_ids)
        fairness_scores = self._compute_fairness_scores(client_ids, server_round)
        
        # Adapt weights over time
        alpha_1, alpha_2 = self._adapt_weights(server_round)
        
        final_scores = {}
        for client_id in client_ids:
            reliability = reliability_scores.get(client_id, 0.0)
            fairness = fairness_scores.get(client_id, 0.0)
            final_scores[client_id] = (alpha_1 * reliability) + (alpha_2 * fairness)
        
        print(f"[Global Score] Round {server_round}: Weights: reliability={alpha_1:.2f}, fairness={alpha_2:.2f}")
        for cid in client_ids[:min(5, len(client_ids))]:
            print(f"  Client {cid}: R={reliability_scores.get(cid, 0):.3f}, F={fairness_scores.get(cid, 0):.3f}, Score={final_scores[cid]:.3f}")
        
        return final_scores

    def _adapt_weights(self, server_round: int) -> Tuple[float, float]:
        if server_round <= self.phase_threshold:
            return 0.7, 0.3
        else:
          return 0.4,0.6

    def _categorize_clients(self, all_client_ids: List[str]) -> Dict[str, List[str]]:
        """Categorize clients based on participation history."""
        categories = {
            'experienced': [],
            'new': [],
            'occasional': []
        }
        
        for client_id in all_client_ids:
            participation_count = self.selection_counts.get(client_id, 0)
            
            if participation_count >= self.min_participation_for_clustering:
                categories['experienced'].append(client_id)
            elif participation_count == 0:
                categories['new'].append(client_id)
            else:
                categories['occasional'].append(client_id)
        
        return categories

    def _bootstrap_selection(self, all_clients: Dict, server_round: int) -> List[Tuple[ClientProxy, FitIns]]:
        """Bootstrap phase: diverse random selection to build client profiles."""
        print(f"[CSMDA] Bootstrap Round {server_round}: Building client diversity")
        
        available_clients = list(all_clients.keys())
        
        # Ensure diversity in bootstrap selection
        selected_clients = []
        
        # Priority 1: Include some new clients
        new_clients = [cid for cid in available_clients if self.selection_counts.get(cid, 0) == 0]
        if new_clients:
            new_selection_count = min(len(new_clients), max(1, self.min_fit_clients // 2))
            selected_clients.extend(random.sample(new_clients, new_selection_count))
        
        # Priority 2: Fill with random selection from remaining
        remaining_needed = self.min_fit_clients - len(selected_clients)
        remaining_clients = [cid for cid in available_clients if cid not in selected_clients]
        
        if remaining_clients and remaining_needed > 0:
            additional_selection = random.sample(
                remaining_clients, 
                min(remaining_needed, len(remaining_clients))
            )
            selected_clients.extend(additional_selection)
        
        print(f"[Bootstrap] Selected {len(selected_clients)} clients: {selected_clients}")
        
        return self._create_fit_instructions(selected_clients, all_clients, server_round)

    def _hybrid_selection(
        self, 
        categories: Dict[str, List[str]], 
        all_clients: Dict, 
        server_round: int
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Hybrid selection combining clustering for experienced clients and opportunities for new ones."""
        
        print(f"[CSMDA] Hybrid Selection Round {server_round}")
        print(f"  - Experienced: {len(categories['experienced'])} clients")
        print(f"  - New: {len(categories['new'])} clients")  
        print(f"  - Occasional: {len(categories['occasional'])} clients")
        
        selected_clients = []
        
        # Stage 1: Core selection from experienced clients using clustering
        if len(categories['experienced']) >= self.num_clusters:
            core_slots = int((1 - self.opportunity_ratio) * self.min_fit_clients)
            core_selection = self._cluster_based_selection(
                categories['experienced'], 
                all_clients, 
                core_slots, 
                server_round
            )
            selected_clients.extend(core_selection)
            print(f"[Hybrid] Core clustering selection: {len(core_selection)} clients")
        
        # Stage 2: Opportunity selection for new clients
        opportunity_slots = int(self.opportunity_ratio * self.min_fit_clients)
        if categories['new'] and opportunity_slots > 0:
            opportunity_count = min(opportunity_slots, len(categories['new']))
            opportunity_selection = random.sample(categories['new'], opportunity_count)
            selected_clients.extend(opportunity_selection)
            print(f"[Hybrid] Opportunity selection: {len(opportunity_selection)} new clients")
        
        # Stage 3: Fill remaining slots
        remaining_slots = self.min_fit_clients - len(selected_clients)
        if remaining_slots > 0:
            remaining_pool = (categories['occasional'] + categories['experienced'] + 
                            [c for c in categories['new'] if c not in selected_clients])
            remaining_candidates = [c for c in remaining_pool if c not in selected_clients]
            
            if remaining_candidates:
                remaining_count = min(remaining_slots, len(remaining_candidates))
                remaining_selection = random.sample(remaining_candidates, remaining_count)
                selected_clients.extend(remaining_selection)
                print(f"[Hybrid] Remaining slots: {len(remaining_selection)} clients")
        
        return self._create_fit_instructions(selected_clients, all_clients, server_round)


    def _cluster_based_selection(
        self, 
        experienced_clients: List[str], 
        all_clients: Dict, 
        num_to_select: int,
        server_round: int
    ) -> List[str]:
        """Apply clustering only to experienced clients with meaningful prototypes."""
        
        print(f"[Clustering] Processing {len(experienced_clients)} experienced clients")
        
        # Collect prototypes only from experienced clients
        prototypes_list = []
        client_ids_with_protos = []
        
        for client_id in experienced_clients:
            client_proxy = all_clients[client_id]
            try:
                get_protos_res = client_proxy.get_properties(
                    ins=GetPropertiesIns(config={"request": "prototypes"}), 
                    timeout=15.0,
                    group_id=None
                )
                
                prototypes_encoded = get_protos_res.properties.get("prototypes")
                
                if prototypes_encoded:
                    prototypes = pickle.loads(base64.b64decode(prototypes_encoded))
                    if isinstance(prototypes, dict) and prototypes:
                        prototypes_list.append(prototypes)
                        client_ids_with_protos.append(client_id)
                        
            except Exception as e:
                print(f"[Clustering] ⚠️ Client {client_id}: {e}")
                continue
        
        print(f"[Clustering] Got prototypes from {len(client_ids_with_protos)} clients")
        
        if len(client_ids_with_protos) >= self.num_clusters:
            # Perform clustering
            if not self.cluster_prototypes:
                print("[Clustering] Initializing clusters")
                self.cluster_prototypes = self._initialize_clusters(prototypes_list)
            
            # Update clustering
            assignments = self._e_step(prototypes_list, client_ids_with_protos)
            self.cluster_prototypes = self._m_step(
                prototypes_list, 
                client_ids_with_protos, 
                assignments
            )
            
            # Update global assignments
            for client_id, cluster_id in assignments.items():
                self.client_assignments[client_id] = cluster_id
            
            # Distribute selection across clusters
            return self._distribute_selection_across_clusters(
                client_ids_with_protos, assignments, num_to_select, server_round
            )
        else:
            print("[Clustering] Not enough clients for clustering, using score-based selection")
            scores = self._compute_global_selection_scores(experienced_clients, server_round)
            sorted_clients = sorted(experienced_clients, key=lambda c: scores.get(c, 0), reverse=True)
            return sorted_clients[:num_to_select]

    def _distribute_selection_across_clusters(
        self, 
        client_ids: List[str], 
        assignments: Dict[str, int], 
        num_to_select: int,
        server_round: int
    ) -> List[str]:
        """Distribute selection quota across clusters based on global scores."""
        
        # Group clients by cluster
        clusters = defaultdict(list)
        for client_id in client_ids:
            cluster_id = assignments.get(client_id, 0)
            clusters[cluster_id].append(client_id)
        
        # Compute selection scores
        global_scores = self._compute_global_selection_scores(client_ids, server_round)
        
        # Distribute selection across active clusters
        active_clusters = list(clusters.keys())
        selected_clients = []
        
        if active_clusters:
            base_per_cluster = max(1, num_to_select // len(active_clusters))
            extra_selections = num_to_select % len(active_clusters)
            
            for i, cluster_id in enumerate(active_clusters):
                cluster_clients = clusters[cluster_id]
                
                # Sort by selection scores
                cluster_clients_sorted = sorted(
                    cluster_clients,
                    key=lambda c: global_scores.get(c, 0),
                    reverse=True
                )
                
                # Select from this cluster
                selections_from_cluster = min(
                    base_per_cluster + (1 if i < extra_selections else 0),
                    len(cluster_clients_sorted)
                )
                
                cluster_selection = cluster_clients_sorted[:selections_from_cluster]
                selected_clients.extend(cluster_selection)
                
                print(f"[Distribution] Cluster {cluster_id}: {len(cluster_selection)} clients selected")
        
        return selected_clients[:num_to_select]

    def _initialize_clusters(self, all_prototypes: List[Dict]) -> Dict:
        """Initialize clusters using K-means++ style approach for better diversity."""
        num_clients = len(all_prototypes)
        assert num_clients >= self.num_clusters
        
        print(f"[Init] Initializing {self.num_clusters} clusters from {num_clients} clients")
        
        cluster_prototypes = {}
        selected_indices = []
        
        # First cluster: random selection
        first_idx = random.randint(0, num_clients - 1)
        selected_indices.append(first_idx)
        
        # Subsequent clusters: select diverse clients
        for cluster_id in range(1, self.num_clusters):
            max_min_dist = -1
            best_idx = 0
            
            for client_idx, proto_dict in enumerate(all_prototypes):
                if client_idx in selected_indices:
                    continue
                    
                # Find minimum distance to existing cluster centers
                min_dist_to_centers = float('inf')
                for existing_idx in selected_indices:
                    dist = self._calculate_client_distance(proto_dict, all_prototypes[existing_idx])
                    min_dist_to_centers = min(min_dist_to_centers, dist)
                
                # Select client with maximum minimum distance
                if min_dist_to_centers > max_min_dist:
                    max_min_dist = min_dist_to_centers
                    best_idx = client_idx
                    
            selected_indices.append(best_idx)
        
        # Initialize cluster prototypes
        for cluster_id, client_idx in enumerate(selected_indices):
            proto_dict = all_prototypes[client_idx]
            cluster_prototypes[cluster_id] = {}
            
            for class_id, proto in proto_dict.items():
                # Ensure numpy array format
                if hasattr(proto, 'numpy'):
                    proto_np = proto.numpy()
                elif hasattr(proto, 'detach'):
                    proto_np = proto.detach().cpu().numpy()
                else:
                    proto_np = np.array(proto)
                
                # Add small noise to break ties
                proto_np = proto_np + np.random.normal(0, 0.01, proto_np.shape)
                cluster_prototypes[cluster_id][class_id] = proto_np.astype(np.float32).copy()

        print(f"[Init] Initialized clusters from clients: {selected_indices}")
        return cluster_prototypes

    def _calculate_client_distance(self, proto_dict1: Dict, proto_dict2: Dict) -> float:
        """Calculate distance between two client prototype dictionaries."""
        total_dist = 0
        shared_classes = 0
        
        for class_id in proto_dict1:
            if class_id in proto_dict2:
                proto1 = np.array(proto_dict1[class_id])
                proto2 = np.array(proto_dict2[class_id])
                dist = self.cosine_distance(proto1, proto2)
                total_dist += dist
                shared_classes += 1
        
        return total_dist / max(1, shared_classes)

    def _e_step(self, all_prototypes: List[Dict], client_ids: List[str]) -> Dict[str, int]:
        """Enhanced E-step with better debugging and tie-breaking."""
        if not self.cluster_prototypes:
            print("[E-step] ERROR: No cluster prototypes available!")
            return {}
        
        assignments = {}
        
        print(f"[E-step] Assigning {len(client_ids)} clients to {len(self.cluster_prototypes)} clusters")
        
        for client_idx, (client_id, prototypes) in enumerate(zip(client_ids, all_prototypes)):
            cluster_distances = {}
            
            for cluster_id in self.cluster_prototypes:
                total_dist = 0
                shared_classes = 0

                for class_id in prototypes:
                    if class_id in self.cluster_prototypes[cluster_id]:
                        client_proto = np.array(prototypes[class_id])
                        cluster_proto = self.cluster_prototypes[cluster_id][class_id]

                        dist = self.cosine_distance(client_proto, cluster_proto)
                        total_dist += dist
                        shared_classes += 1

                # Average distance with coverage penalty
                if shared_classes > 0:
                    avg_dist = total_dist / shared_classes
                    coverage_penalty = 1.0 - (shared_classes / len(prototypes))
                    avg_dist += 0.1 * coverage_penalty
                else:
                    avg_dist = 1.0
                    
                cluster_distances[cluster_id] = avg_dist

            # Find best cluster with tie-breaking
            sorted_clusters = sorted(cluster_distances.items(), key=lambda x: (x[1], x[0]))
            best_cluster = sorted_clusters[0][0]
            assignments[client_id] = best_cluster
            
            # Debug for first few clients
            if client_idx < 3:
                distances_str = {k: f"{v:.4f}" for k, v in cluster_distances.items()}
                print(f"[E-step] Client {client_id}: {distances_str} → Cluster {best_cluster}")

        # Summary and warning
        cluster_counts = defaultdict(int)
        for cluster_id in assignments.values():
            cluster_counts[cluster_id] += 1
        
        print(f"[E-step] Assignment summary: {dict(cluster_counts)}")
        
        if len(set(assignments.values())) == 1:
            print("[E-step] ⚠️ WARNING: All clients assigned to same cluster!")
            # Force diversity
            if len(assignments) >= self.num_clusters:
                client_list = list(assignments.keys())
                for i in range(min(self.num_clusters, len(client_list))):
                    assignments[client_list[i]] = i
                print(f"[E-step] Applied forced diversity")
        
        return assignments

    def _m_step(self, all_prototypes: List[Dict], client_ids: List[str], assignments: Dict[str, int]) -> Dict:
        """M-step: Update cluster prototypes."""
        new_cluster_prototypes = {}
        
        for cluster_id in range(self.num_clusters):
            cluster_clients = [cid for cid in client_ids if assignments.get(cid) == cluster_id]
            
            if cluster_clients:
                # Collect prototypes for this cluster
                cluster_proto_dicts = []
                for client_id in cluster_clients:
                    client_idx = client_ids.index(client_id)
                    cluster_proto_dicts.append(all_prototypes[client_idx])
                
                # Average prototypes by class
                new_cluster_prototypes[cluster_id] = {}
                all_classes = set()
                for proto_dict in cluster_proto_dicts:
                    all_classes.update(proto_dict.keys())
                
                for class_id in all_classes:
                    class_protos = []
                    for proto_dict in cluster_proto_dicts:
                        if class_id in proto_dict:
                            proto = np.array(proto_dict[class_id])
                            class_protos.append(proto)
                    
                    if class_protos:
                        avg_proto = np.mean(class_protos, axis=0)
                        new_cluster_prototypes[cluster_id][class_id] = avg_proto.astype(np.float32)
            else:
                # Keep old prototype if no clients assigned
                if cluster_id in self.cluster_prototypes:
                    new_cluster_prototypes[cluster_id] = self.cluster_prototypes[cluster_id]
        
        return new_cluster_prototypes

    def cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Improved cosine distance with numerical stability."""
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0
        
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        similarity = np.clip(similarity, -1.0, 1.0)
        
        return 1.0 - similarity

    def _create_fit_instructions(
        self, 
        selected_client_ids: List[str], 
        all_clients: Dict, 
        server_round: int
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Create FitIns for selected clients."""
        instructions = []
        
        for client_id in selected_client_ids:
            if client_id in all_clients:
                client_proxy = all_clients[client_id]
                client_config = {
                    "server_round": server_round,
                    "total_rounds": getattr(self, 'total_rounds', 100),
                }
                
                instructions.append((client_proxy, FitIns(self.current_parameters, client_config)))
                
                # Update selection count
                self.selection_counts[client_id] = self.selection_counts.get(client_id, 0) + 1
        
        return instructions

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Literature-inspired progressive client selection strategy.
        """
        print(f"\n[CSMDA] ========== Round {server_round} Configuration ==========")
        
        self.current_parameters = parameters
        all_clients = client_manager.all()
        available_client_ids = list(all_clients.keys())

        if not available_client_ids:
            print(f"[CSMDA] No clients available for round {server_round}")
            return []

        print(f"[CSMDA] Available clients: {len(available_client_ids)}")
        
        # Categorize clients by experience level
        categories = self._categorize_clients(available_client_ids)
        
        print(f"[CSMDA] Client categories:")
        for category, clients in categories.items():
            print(f"  - {category.capitalize()}: {len(clients)} clients")

        # Selection strategy based on system maturity
        if (server_round <= self.bootstrap_rounds or 
            len(categories['experienced']) < self.num_clusters):
            # Bootstrap phase
            instructions = self._bootstrap_selection(all_clients, server_round)
        else:
            # Mature phase with hybrid selection
            instructions = self._hybrid_selection(categories, all_clients, server_round)

        # Statistics
        self.round_stats[server_round] = {
            'total_available': len(available_client_ids),
            'selected': len(instructions),
            'categories': {k: len(v) for k, v in categories.items()},
            'selection_strategy': 'bootstrap' if server_round <= self.bootstrap_rounds else 'hybrid'
        }

        selected_cids = [inst[0].cid for inst in instructions]
        print(f"[CSMDA] ✅ Round {server_round} Final Selection:")
        print(f"  - Selected: {len(instructions)} clients: {selected_cids}")
        print(f"  - Strategy: {self.round_stats[server_round]['selection_strategy']}")
        if hasattr(self, 'client_assignments') and self.client_assignments:
            print(f"  - Cluster assignments: {dict(self.client_assignments)}")
        print(f"[CSMDA] ========== End Round {server_round} ==========\n")

        return instructions

    def configure_evaluate(
      self, server_round: int, parameters: Parameters, client_manager: ClientManager
) -> List[Tuple[ClientProxy, EvaluateIns]]:
           
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
      

  

