import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np
from torch.distributions import Dirichlet, Categorical

def update_global_model(self, classifier_params: List[List[np.ndarray]]) -> None:
    """Aggregate client classifier parameters using FedAvg and update global_model."""
    if not classifier_params:
        raise ValueError("classifier_params is empty")
    
    # Number of clients
    num_clients = len(classifier_params)
    
    # Initialize aggregated parameters
    # Assume all clients have the same model architecture
    aggregated_params = [
        np.mean([params[i] for params in classifier_params], axis=0)
        for i in range(len(classifier_params[0]))
    ]
    
    # Create or update global_model
    self.global_model = self._create_client_model(aggregated_params)
    self.global_model.to(self.device)
    self.global_model.eval()

def _train_generator(self, classifier_ids, label_probs, classifier_params: List[List[np.ndarray]]):
    with self.mlflow.start_run(run_id=self.server_run_id):
        """Train the generator using the ensemble of client classifiers with Dirichlet sampling and global model distillation."""
        noise_dim = 64
        label_dim = 2
        hidden_dim = 256
        output_dim = 64
        batch_size = 13
        learning_rate = 0.001
        num_epochs = 15
        num_domains = 3
        lambda_kl = 0.3  # KL divergence weight
        lambda_adv = 0.28  # Adversarial loss weight
        alpha = 0.5  # Dirichlet concentration parameter
        temperature = 2.0  # Temperature for softening global model predictions

        # Update global_model using FedAvg
        self.update_global_model(classifier_params)

        # Optimizers
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        bce_loss = nn.BCELoss()

        for epoch in range(num_epochs):
            print('====== Training Generator =====')
            epoch_loss = 0.0

            # Sample labels from Dirichlet distribution
            dirichlet = Dirichlet(torch.full((label_dim,), alpha).to(self.device))
            pi = dirichlet.sample()  # Shape: [label_dim]
            categorical = Categorical(pi)
            labels = categorical.sample((batch_size,)).to(self.device)  # Shape: [batch_size]
            labels_one_hot = F.one_hot(labels, num_classes=label_dim).float().to(self.device)

            # Sample noise
            noise = torch.randn(batch_size, noise_dim, dtype=torch.float32).to(self.device)

            # Train Discriminator
            optimizer_d.zero_grad()
            z, mu, logvar = self.generator(noise, labels_one_hot, return_distribution=True)
            real_z = torch.randn(batch_size, output_dim).to(self.device)
            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)
            d_loss = 0.5 * (bce_loss(self.discriminator(real_z), real_labels) +
                            bce_loss(self.discriminator(z.detach()), fake_labels))
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer.zero_grad()
            z, mu, logvar = self.generator(noise, labels_one_hot, return_distribution=True)
            logits = []

            # Compute client logits
            for params in classifier_params:
                if hasattr(params, '__iter__') and not isinstance(params, (list, tuple, np.ndarray)):
                    params = [p.detach().cpu().numpy() for _, p in params]
                
                client_model = self._create_client_model(params)
                client_model = client_model.to(self.device)
                client_model.eval()
                
                client_logits = client_model(z)
                logits.append(client_logits)
            
            if not logits:
                raise ValueError("No valid logits generated from client models")
            
            # Average the client logits
            avg_logits = torch.mean(torch.stack(logits), dim=0)

            # Compute global model logits
            global_logits = self.global_model(z)
            global_probs = F.softmax(global_logits / temperature, dim=1)  # Apply temperature scaling

            # Compute losses
            distill_loss = criterion(avg_logits, global_probs)  # Distillation loss with global model
            kl_loss = self.kl_divergence(mu, logvar)
            adv_loss = bce_loss(self.discriminator(z), real_labels)
            
            # Total loss
            total_loss = distill_loss + lambda_kl * kl_loss + lambda_adv * adv_loss

            # Backpropagate and update generator
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

            # Log metrics using mlflow
            self.mlflow.log_metrics({
                "generator_loss": epoch_loss,
                "distill_loss": distill_loss.item(),
                "kl_loss": kl_loss.item(),
                "adv_loss": adv_loss.item(),
                "epoch": epoch,
            }, step=epoch)

            print(f"Generator Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            print(f"kl loss Epoch [{epoch + 1}/{num_epochs}], Kl Loss: {kl_loss:.4f}")
            print(f"distill loss Epoch [{epoch + 1}/{num_epochs}], distill Loss: {distill_loss:.4f}")
            print(f"adv loss Epoch [{epoch + 1}/{num_epochs}], adv Loss: {adv_loss:.4f}")