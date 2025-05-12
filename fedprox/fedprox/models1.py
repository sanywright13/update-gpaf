"""CNN model architecture, training, and testing functions for MNIST."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.autograd import Variable
#GLOBAL Generator 

# use a Generator Network with reparametrization trick
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from models.swin_transformer import SwinTransformer
#model vit
#from vit_pytorch.vit_for_small_dataset import ViT
import sys
import os

# Get the path to the nested repo relative to your current script
nested_repo_path = os.path.join(os.path.dirname(__file__), "..","..", "..","Swin-Transformer-fed")
sys.path.append(os.path.abspath(nested_repo_path))
print(f'gg: {nested_repo_path}')
from models.swin_transformer import SwinTransformer
from models.swin_transformer import PatchEmbed, SwinTransformerBlock  # Import from your SWIM file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
import torch
def get_model(model_name):
  if model_name == 'vit':
    model = ViT(
    image_size=28,        # specify image size
    patch_size=14,
    num_classes=2,        # specify the number of output classes
    dim=128,               # embedding dimension
    depth=8,               # number of transformer layers
    heads=4,               # number of attention heads
    mlp_dim=512,          # MLP hidden layer dimension
    pool='mean',            # 'cls' or 'mean' pooling
    channels=1,            # number of input channels (e.g., 3 for RGB images)
    dim_head=64,           # dimension per attention head
    dropout=0.3,
    #emb_dropout=0.1        # embedding dropout rate
    ).to(device)
  elif model_name == 'swim':
    layernorm = nn.LayerNorm
    USE_CHECKPOINT=False
    FUSED_WINDOW_PROCESS=False
    IMG_SIZE=28
    IN_CHANS=1
    NUM_CLASSES=2
    DEPTHS= [4,6]
    NUM_HEADS=[12,24]
    WINDOW_SIZE=7
    MLP_RATIO=4
    PATCH_SIZE=2
    EMBED_DIM=96
    QKV_BIAS=True
    QK_SCALE=None
    DROP_RATE=0.1
    DROP_PATH_RATE=0.2
    APE=False
    PATCH_NORM=True
    model = SwinTransformer(img_size=IMG_SIZE,
                                patch_size=PATCH_SIZE,
                                in_chans=IN_CHANS,
                                num_classes=NUM_CLASSES,
                                embed_dim=EMBED_DIM,
                                depths=DEPTHS,
                                num_heads=NUM_HEADS,
                                window_size=WINDOW_SIZE,
                                mlp_ratio=MLP_RATIO,
                                qkv_bias=QKV_BIAS,
                                qk_scale=QK_SCALE,
                                drop_rate=DROP_RATE,
                                drop_path_rate=DROP_PATH_RATE,
                                ape=APE,
                                norm_layer=layernorm,
                                patch_norm=PATCH_NORM,
                                use_checkpoint=USE_CHECKPOINT,
                                fused_window_process=FUSED_WINDOW_PROCESS)

  return model
Tensor = torch.FloatTensor
# First, let's define the GRL layer for client side

class GlobalGenerator(nn.Module):
    def __init__(self, noise_dim, label_dim, hidden_dim, output_dim):
        super().__init__()
        self.noise_dim = noise_dim
        self.label_dim = label_dim
        
        # Initial projection for noise
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Initial projection for labels
        self.label_proj = nn.Sequential(
            nn.Linear(label_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Mu and logvar projections
        self.mu_proj = nn.Linear(2 * hidden_dim, output_dim)
        self.logvar_proj = nn.Linear(2 * hidden_dim, output_dim)
        
        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, noise, labels, return_distribution=False):
        # Project noise and labels to same dimension
        noise_feat = self.noise_proj(noise)  # [batch_size, hidden_dim]
        label_feat = self.label_proj(labels)  # [batch_size, hidden_dim]
        
        # Combine features
        combined = torch.cat([noise_feat, label_feat], dim=1)  # [batch_size, 2*hidden_dim]
        
        # Generate mu and logvar
        mu = self.mu_proj(combined)
        logvar = self.logvar_proj(combined)
        
        # Apply reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Final output projection
        features = self.output_proj(z)
        
        if return_distribution:
            return features, mu, logvar
        return features





class GradientReversalFunction(torch.autograd.Function):
    """
    Custom autograd function for gradient reversal.
    Forward: Acts as identity function
    Backward: Reverses gradient by multiplying by -lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        # Store lambda for backward pass
        ctx.lambda_ = lambda_
        # Forward pass is identity function
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse gradient during backward pass
        # grad_output: gradient from subsequent layer
        # -lambda * gradient gives us gradient reversal
        return ctx.lambda_ * grad_output.neg(), None

class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer.
    Implements gradient reversal for adversarial training.
    """
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
        
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
        


class ServerDiscriminator(nn.Module):
    def __init__(self, feature_dim, num_domains):
        super().__init__()
        # Gradient Reversal Layer
        self.grl = GradientReversalLayer()
        # Discriminator architecture
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, num_domains)
        )

    def forward(self, x, lambda_=1.0):
        self.grl.lambda_ = lambda_
        # Apply GRL before discrimination
        x = self.grl(x)
        return self.discriminator(x)

def reparameterize(mu, logvar):

    std = torch.exp(0.5 * logvar)  # Standard deviation
    eps = torch.randn_like(std)    # Random noise from N(0, I)
    z = mu + eps * std             # Reparameterized sample
    return z
def sample_labels(batch_size, label_probs):
  
    #print(f'lqbel prob {label_probs}')
    # Extract probabilities from the dictionary
    probabilities = list(label_probs.values())
    
    # Extract labels from the dictionary
    labels = list(label_probs.keys())
    sampled_labels = np.random.choice(labels, size=batch_size, p=probabilities)
    return torch.tensor(sampled_labels, dtype=torch.long)

def generate_feature_representation(generator, noise, labels_one_hot):
   
    z = generator(noise, labels_one_hot)
    return z
#in our GPAF we will train a VAE-GAN local model in each client
img_shape=(28,28)
def reparameterization(mu, logvar,latent_dim):
    std = torch.exp(logvar / 2)
    #sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    sampled_z = torch.randn_like(mu)  # Sample from standard normal distribution
    z = sampled_z * std + mu
    return z
'''
class Encoder(nn.Module):
    def __init__(self,latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim=latent_dim
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, img):
        #print(f"Encoder input shape (img): {img.shape}")  # Debug: Print input shape

        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        #print(f"Encoder model output shape (x): {x.shape}")  # Debug: Print model output shape

        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar,self.latent_dim)
        #print(f"Encoder output shape (z): {z.shape}")  # Debug: Print output shape

        #self._register_hooks()
        return z
'''        


class Discriminator(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        # No sigmoid for WGAN
        return self.model(x)

# Values from your FedAvg architecture:
QKV_BIAS=True
DROP_RATE=0.1
DROP_PATH_RATE=0.2
PATCH_NORM=True



class Encoder(SwinTransformer):
    def __init__(self, latent_dim=64):
        # Initialize parent SwinTransformer with all parameters
        self.latent_dim = latent_dim
        super().__init__(
            img_size=28,
            patch_size=2,
            in_chans=1,
            num_classes=0,
            embed_dim=96,
            depths=[4,6],
            num_heads=[12,24],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.1,
            drop_path_rate=0.2,
            ape=False,
            patch_norm=True
        )
        # Enhanced projection head
        self.projection = nn.Sequential(
            nn.LayerNorm(768),  # Normalize features
            nn.Linear(768, 512),
            nn.GELU(),  # GELU typically works better with transformers
            nn.Dropout(0.1),
            nn.Linear(512, self.latent_dim * 2)
        )
        
       
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # During evaluation, just return mean
    
    def forward(self, x):
        # Get Swin features
        features = self.swin.forward_features(x)
        
        # Project to latent parameters
        latent_params = self.projection(features)
        mu, logvar = torch.chunk(latent_params, 2, dim=-1)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar  # Return all for computing KL divergence

class Classifier(nn.Module):
    def __init__(self, num_features=192, num_classes=2):  # 384 = 96 * 2^2
        super().__init__()
        self.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.head(x)

class Discriminator(nn.Module):
    def __init__(self,latent_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        # Register hooks to track shapes
        #self._register_hooks()

    def forward(self, z):
        validity = self.model(z)
        return validity
    

        # Register hooks for each layer in self.model
        for layer in self.model:
            layer.register_forward_hook(hook_fn)
'''
class Classifier(nn.Module):
    def __init__(self,latent_dim,num_classes=2):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_classes),  # Output layer for multi-class classification
      
        )

    def forward(self, z):
        logits = self.model(z)
        return logits
 


'''
# Add these helper functions for monitoring training
def compute_wasserstein_distance(discriminator, real_features, fake_features):
    """Compute Wasserstein distance between real and fake features"""
    with torch.no_grad():
        d_real = discriminator(real_features).mean()
        d_fake = discriminator(fake_features).mean()
        return (d_real - d_fake).item()

def compute_gradient_penalty(discriminator, real_features, fake_features, device):
    """Compute gradient penalty for improved WGAN training"""
    batch_size = real_features.size(0)
    alpha = torch.rand(batch_size, 1).to(device)
    alpha = alpha.expand_as(real_features)
    
    # Interpolated features
    interpolated = alpha * real_features + (1 - alpha) * fake_features
    interpolated.requires_grad_(True)
    
    # Discriminator output for interpolated features
    d_interpolated = discriminator(interpolated)
    
    # Compute gradients
    grad_outputs = torch.ones_like(d_interpolated).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Compute gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty
def train_gpaf( encoder: nn.Module,
classifier,
discriminator,
    trainloader: DataLoader,
    device: torch.device,
    client_id,
    epochs: int,
   global_generator,server_discriminator
    ):

# 
    learning_rate=0.01
    
    #global_params = [val.detach().clone() for val in net.parameters()]
    
    net = train_one_epoch_gpaf(
        encoder,
classifier,discriminator , trainloader, device,client_id,
            epochs,global_generator,server_discriminator
        )

class AdaptiveGradientPenalty:
    def __init__(self, 
                 initial_lambda=10.0,
                 target_gp=1.0,
                 adaptation_rate=0.01,
                 min_lambda=1.0,
                 max_lambda=50.0):
        self.lambda_gp = initial_lambda
        self.target_gp = target_gp
        self.adaptation_rate = adaptation_rate
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.moving_avg_gp = None
        
    def update(self, current_gp):
        """Update lambda_gp based on the current gradient penalty value"""
        # Initialize moving average
        if self.moving_avg_gp is None:
            self.moving_avg_gp = current_gp
        
        # Update moving average
        self.moving_avg_gp = 0.9 * self.moving_avg_gp + 0.1 * current_gp
        
        # Compute error from target
        error = self.moving_avg_gp - self.target_gp
        
        # Update lambda_gp
        self.lambda_gp = self.lambda_gp + self.adaptation_rate * error
        
        # Clip to valid range
        self.lambda_gp = max(self.min_lambda, min(self.max_lambda, self.lambda_gp))
        
        return self.lambda_gp

#we must add a classifier that classifier into a binary categories
#send back the classifier parameter to the server
def train_one_epoch_gpaf(encoder,classifier,discriminator,trainloader, DEVICE,client_id, epochs,global_generator,server_discriminator=None,verbose=False):
    """Train the network on the training set."""
    #print(f'local global representation z are {global_z}')
    #criterion = torch.nn.CrossEntropyLoss()
    lr=0.00013914064388085564
    #optimizer = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=1e-4)

    # Initialize adaptive gradient penalty
    gp_handler = AdaptiveGradientPenalty(
        initial_lambda=10.0,
        target_gp=1.0,
        adaptation_rate=0.01
    )
    
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-4)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_C = torch.optim.Adam(classifier.parameters(), lr=lr,weight_decay=1e-4)
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    criterion_cls = nn.CrossEntropyLoss()  # Classification loss (for binary classification)
    for epoch in range(epochs):
        print('==start local training ==')
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch
            images, labels = images.to(DEVICE , dtype=torch.float32), labels.to(DEVICE  , dtype=torch.long)
            
           
            # Ensure labels have shape (N,1)
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(1)
            else:
                  labels=labels.squeeze(1)
            
            real_imgs = images.to(DEVICE)

            # Generate global z representation
            batch_size = 13
            #self.noise_dim = self.latent_dim - self.num_classes  # 192 - 2 = 190

            noise = torch.randn(batch_size, 62, dtype=torch.float32).to(DEVICE)
            labels_onehot = F.one_hot(labels.long(), num_classes=2).float()
            #print(f'real_imgs eee ftrze{labels_onehot.shape} and {noise.shape}')
            noise = torch.tensor(noise, dtype=torch.float32)
            with torch.no_grad():
                    
              global_z = global_generator(noise, labels_onehot)
            # ---------------------
            # Train Discriminator
            # ---------------------
            #optimizer_D.zero_grad()

            # Real loss: Discriminator should classify global z as 1
            
            if global_z is not None:
                    real_labels = torch.ones(global_z.size(0), 1, device=DEVICE, dtype=torch.float32)  # Real labels
                    #print(f' z shape on train {real_labels.shape}')
                    real_loss = criterion(discriminator(global_z), real_labels)
                    #print(f' dis glob z shape on train {discriminator(global_z).shape}')

            else:
                    real_loss = 0

            # Fake loss: Discriminator should classify local features as 0
            #local_features = encoder(real_imgs)
            
            
            # Fake loss: Discriminator should classify local features as 0
            #local_features = encoder(real_imgs)
            #fake_labels = torch.zeros(real_imgs.size(0), 1 , dtype=torch.float32)  # Fake labels
            #fake_loss = criterion(discriminator(local_features.detach()), fake_labels)
            #print(f'local train feat {discriminator(local_features.detach()).shape}')
            #print(f'local encoder features {local_features}')
            
            # Total discriminator loss
            '''
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            optimizer_D.step()
            '''
            # ---------------------
            #  Train Discriminator
            # ---------------------
            for _ in range(5):  # Multiple D updates per G update
                optimizer_D.zero_grad()
                
                # Extract local features
                local_features = encoder(real_imgs)
                
                # Real loss (global features)
                d_real = discriminator(global_z.detach())
                
                # Fake loss (local features)
                d_fake = discriminator(local_features.detach())
                # Get adaptive lambda_gp
                # Gradient penalty
                gp = compute_gradient_penalty(
                    discriminator,
                    global_z.detach(),
                    local_features.detach(),
                    DEVICE
                )
                current_lambda = gp_handler.update(gp.item())
                # Wasserstein loss
                d_loss = torch.mean(d_fake) - torch.mean(d_real)
                
                
                
                # Total discriminator loss
                d_total_loss = d_loss + current_lambda * gp
                d_total_loss.backward()
                optimizer_D.step()
            

            # -----------------
            # Train Generator
            # -----------------
            optimizer_E.zero_grad()
            
            
            # Get fresh features for encoder training
            #local_features = encoder(images)
            # Feature matching loss
            feature_matching_loss = F.mse_loss(
                local_features.mean(0), global_z.mean(0)
            ) + 0.5 * F.mse_loss(
                local_features.std(0), global_z.std(0)
            )
            
            #g_loss = criterion(discriminator(local_features), real_labels)
            # Adversarial loss (WGAN)
            g_loss = -torch.mean(discriminator(local_features))+ 0.1 * feature_matching_loss 
            #local minimizing federated adverserial loss

            
            encoder_loss = g_loss
        
            encoder_loss.backward()
            optimizer_E.step()

            #Classification loss with label
            
            #print(f' label size shape {labels.shape}')
            # -----------------
            # Train Classifier
            # -----------------
            optimizer_C.zero_grad()

            # Classification loss
            logits = classifier(local_features.detach())  # Detach to avoid affecting encoder
            cls_loss = criterion_cls(logits, labels)
            cls_loss.backward()
            optimizer_C.step()

            
            # Compute accuracy
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Accumulate loss
            epoch_loss += cls_loss.item()
            #print(labels)
            
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        
        print(f"local Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f} (Client {client_id})")
        #print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc} of client : {client_id}")

def test_gpaf(encoder,classifier, testloader,device):
        """Evaluate the network on the entire test set."""
        encoder.eval()
        classifier.eval()

        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        print(f' ==== client test func')
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device , dtype=torch.float32), labels.to(device ,dtype=torch.long)
                labels=labels.squeeze(1)
                # Forward pass
                features = encoder(inputs)
                outputs = classifier(features)

                # Compute loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Compute average loss and accuracy
        avg_loss = total_loss / len(testloader)
        avg_accuracy = correct / total

        return avg_loss, avg_accuracy

def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Evaluate the network on the entire test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    device : torch.device
        The device on which the model should be tested, either 'cpu' or 'cuda'.

    Returns
    -------
    Tuple[float, float]
        The loss and the accuracy of the input model on the given data.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy
