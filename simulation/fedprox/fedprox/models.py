"""CNN model architecture, training, and testing functions for MNIST."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler
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
nested_repo_path = os.path.join(os.path.dirname(__file__), "..", "..", "..","Swin-Transformer-fed")
sys.path.append(os.path.abspath(nested_repo_path))
print(f'gg: {nested_repo_path}')
from models.swin_transformer import SwinTransformer
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

  elif model_name =='resnet':
        pass
  return model
Tensor = torch.FloatTensor
# First, let's define the GRL layer for client side

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
        
class GlobalGenerator(nn.Module):
    def __init__(self, noise_dim, label_dim, domain_dim, hidden_dim, output_dim, num_domains=3):
        super().__init__()
        self.noise_dim = noise_dim
        self.label_dim = label_dim
        self.domain_dim = domain_dim
        
        # Domain embedding layer
        self.domain_embedding_layer = nn.Embedding(num_domains, domain_dim)
        
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
        
        # Initial projection for domain embeddings
        self.domain_proj = nn.Sequential(
            nn.Linear(domain_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Combined feature processing
        self.combined_proj = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
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
    
    def forward(self, noise, labels, domain_indices):
        # Get domain embeddings from indices
        # Ensure domain_indices are long type
        domain_indices = domain_indices.long()
        domain_embeddings = self.domain_embedding_layer(domain_indices)
        
        # Project each input to same dimension
        noise_feat = self.noise_proj(noise)  # [batch_size, hidden_dim]
        label_feat = self.label_proj(labels)  # [batch_size, hidden_dim]
        domain_feat = self.domain_proj(domain_embeddings)  # [batch_size, hidden_dim]
        
        # Combine all features
        combined = torch.cat([noise_feat, label_feat, domain_feat], dim=1)
        
        # Process combined features
        processed = self.combined_proj(combined)
        
        # Generate mu and logvar
        mu = self.mu_proj(processed)
        logvar = self.logvar_proj(processed)
        
        # Apply reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Final output projection
        features = self.output_proj(z)
        
        return features




class ServerDiscriminator(nn.Module):
    def __init__(self, feature_dim, num_domains):
        super().__init__()
        # Remove GRL and use standard discriminator architecture
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, num_domains),
            nn.LogSoftmax(dim=1)  # Use LogSoftmax for numerical stability
        )

    def forward(self, x):
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


#replace shalow feature extractor with swim architecture 
'''
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
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
        # Replace sequential model with Swin Transformer
        self.swin = SwinTransformer(img_size=IMG_SIZE,
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
  
   
        # Remove the classification head
        delattr(self.swin, 'head')
        
        # Add a feature processing layer (similar to your original 512 dims)
        self.feature_process = nn.Sequential(
            nn.Linear(self.swin.num_features, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512)
        )
        
        # Keep the same mu and logvar projections
        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, img):
        # Get Swin features
        features = self.swin.forward_features(img)
        
        # Process features
        x = self.feature_process(features)
        
        # Get mu and logvar
        mu = self.mu(x)
        logvar = self.logvar(x)
        
        # Sample using reparameterization
        z = reparameterization(mu, logvar, self.latent_dim)
        
        return z
'''
#resnet for fedavg

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetBreastMNIST(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super().__init__()
        
        # Initial channel is 1 for grayscale images
        self.inplanes = 32  # Reduced from 64 to handle smaller images
        
        # First conv layer modified for 28x28 grayscale input
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        # Main layers
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet18_breastmnist():
    """ResNet-18 model adapted for BreastMNIST dataset"""
    return ResNetBreastMNIST(BasicBlock, [2, 2, 2, 2])


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
    
        
    

class LocalDiscriminator(nn.Module):
    """Modified discriminator for multi-domain classification."""
    def __init__(self, feature_dim, num_domains):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 3)  # Output logits for each domain
        )
    
    def forward(self, x):
        return self.model(x)

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
    def _register_hooks(self):
        """Register hooks to track shapes at each layer."""
        def hook_fn(module, input, output):
            print(f"Layer: {module.__class__.__name__}")
            print(f"Input shape: {input[0].shape}")
            print(f"Output shape: {output.shape}")
            print("-" * 20)

        # Register hooks for each layer in self.model
        for layer in self.model:
            layer.register_forward_hook(hook_fn)

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
        self._register_hooks()

    def forward(self, z):
        logits = self.model(z)
        return logits
    def _register_hooks(self):
        """Register hooks to track shapes at each layer."""
        def hook_fn(module, input, output):
            print(f"Layer: {module.__class__.__name__}")
            print(f"Input shape: {input[0].shape}")
            print(f"Output shape: {output.shape}")
            print("-" * 20)
#for moon model 

class ModelMOON(nn.Module):
    """Model for MOON."""

    def __init__(self,out_dim, n_classes):
        super().__init__()

        basemodel = resnet18_breastmnist()
        self.features = nn.Sequential(*list(basemodel.children())[:-1])
        num_ftrs = basemodel.fc.in_features
      

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

        # last layer
        self.l3 = nn.Linear(out_dim, n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            return model
        except KeyError as err:
            raise ValueError("Invalid model name.") from err

    def forward(self, x):
        """Forward."""
        h = self.features(x)
        h = h.squeeze()
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

        y = self.l3(x)
        return h, x, y



#contrastive loss for gpaf
def contrastive_loss(local_features, global_features, temperature=0.5):
   cos = torch.nn.CosineSimilarity(dim=-1)
   
   # Local-global alignment (positive pairs)
   positive_sim = cos(local_features, global_features)
   positive_loss = torch.mean(1 - positive_sim)
   
   # Feature diversity (negative pairs)
   batch_size = local_features.size(0)
   feature_sims = cos(local_features.unsqueeze(1), local_features.unsqueeze(0))
   # Remove diagonal (self-similarity)
   mask = ~torch.eye(batch_size, dtype=torch.bool, device=local_features.device)
   negative_loss = torch.mean(feature_sims[mask])
   
   return positive_loss - temperature * negative_loss
def train_gpaf( encoder: nn.Module,
classifier,
discriminator,
    trainloader: DataLoader,
    device: torch.device,
    client_id,
    epochs: int,
   global_generator,domain_discriminator
    ):

# 
    learning_rate=0.01
        
    grads = train_one_epoch_gpaf(
        encoder,
classifier,discriminator , trainloader, device,client_id,
            epochs,global_generator,domain_discriminator
        )
    return grads
  
#we must add a classifier that classifier into a binary categories
#send back the classifier parameter to the server
def train_one_epoch_gpaf(encoder,classifier,discriminator,trainloader, DEVICE,client_id, epochs,global_generator,local_discriminator,verbose=False):
    """Train the network on the training set."""
    #criterion = torch.nn.CrossEntropyLoss()
    lr=0.00013914064388085564
    num_clients=3
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-4)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_C = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    criterion_cls = nn.CrossEntropyLoss()  # Classification loss (for binary classification)
    encoder.train()
    classifier.train()
    discriminator.train()
    local_discriminator.train()
    for epoch in range(epochs):
        print('==start local training ==')
        correct, total, epoch_loss ,loss_sumi ,loss_sum = 0, 0, 0.0 , 0 , 0
        for batch_idx, batch in enumerate(trainloader):
            images, labels = batch
            images, labels = images.to(DEVICE , dtype=torch.float32), labels.to(DEVICE  , dtype=torch.long)
            
            lambda_align = 1.0   # Full weight to alignment loss
            lambda_adv = 0.1     # Start smaller for adversarial component
            # Ensure labels have shape (N,1)
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(1)
            else:
                  labels=labels.squeeze(1)
            
            real_imgs = images.to(DEVICE)

            # Generate global z representation
            batch_size = 13
            noise = torch.randn(batch_size, 64, dtype=torch.float32).to(DEVICE)
            labels_onehot = F.one_hot(labels.long(), num_classes=2).float()
            #print(f'real_imgs eee ftrze{labels_onehot.shape} and {noise.shape}')
            noise = torch.tensor(noise, dtype=torch.float32)
            domain_indices = torch.full((batch_size,), client_id, device=DEVICE, dtype=torch.long)  # Fixed dtype

            # Create domain embedding for current client

            with torch.no_grad():
              global_z = global_generator(noise, labels_onehot.to(DEVICE), domain_indices)
            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real loss: Discriminator should classify global z as 1
            
            if global_z is not None:
                    real_labels = torch.ones(global_z.size(0), 1, device=DEVICE, dtype=torch.float32)  # Real labels
                    #print(f' z shape on train {real_labels.shape}')
                    real_loss = criterion(discriminator(global_z), real_labels)
                    #print(f' dis glob z shape on train {discriminator(global_z).shape}')

            else:
                    real_loss = 0

            # Fake loss: Discriminator should classify local features as 0
            local_features = encoder(real_imgs)
            
            
            # Fake loss: Discriminator should classify local features as 0
            #local_features = encoder(real_imgs)
            fake_labels = torch.zeros(real_imgs.size(0), 1 , dtype=torch.float32)  # Fake labels
            fake_loss = criterion(discriminator(local_features.detach()), fake_labels)
           
            # Total discriminator loss
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            # Train Generator
            # -----------------
             # 3. Train Encoder and Classifier
            optimizer_E.zero_grad()
            optimizer_C.zero_grad()
             
            # Get fresh features for encoder training
            local_features = encoder(images)
            local_features.requires_grad_(True)
            g_loss = criterion(discriminator(local_features), real_labels)

        
            # a) Alignment loss - make local features match global distribution
            local_features = encoder(images)
          
            # b) Domain confusion loss with GRL
            grl_features = GradientReversalLayer()(local_features)
            confusion_logits = local_discriminator(grl_features)
            
            # Create uniform distribution target
            uniform_target = torch.full(
                (batch_size, num_clients), 
            1.0/num_clients,
            device=device
             )
        
            # KL divergence for domain confusion
            confusion_loss = F.kl_div(
            F.log_softmax(confusion_logits, dim=1),
            uniform_target,
            reduction='batchmean'
            )
            # Add contrastive loss
            contrast_loss = contrastive_loss(local_features, global_z, temperature=0.5)
        
            # Combine losses
            lambda_confusion = 1.0
            lambda_contrast = 0.9
            loss = lambda_confusion * confusion_loss + lambda_contrast * contrast_loss
            
            
            
            #loss_sumi += loss_sum.item()
            grads = torch.autograd.grad(
                loss, list(local_discriminator.parameters()), create_graph=True, retain_graph=True
            )
            alpha=0.0002
            for param, grad_ in zip(local_discriminator.parameters(), grads):
                param.data = param.data - alpha * grad_

            for param in local_discriminator.parameters():
                if param.grad is not None:
                    param.grad.zero_()

                    
            # Classification loss
            logits = classifier(local_features)  # Detach to avoid affecting encoder
            cls_loss = criterion_cls(logits, labels)

            # Total loss for encoder
            total_loss = cls_loss +g_loss 
           
            total_loss.backward()
            
            optimizer_E.step()
            optimizer_C.step()
          
             # Accumulate loss
            epoch_loss += total_loss.item()
            loss += loss * labels.size(0)
            # Compute accuracy
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            

        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        
        print(f"local Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f} (Client {client_id})")
        #print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc} of client : {client_id}")
    
    

    for param in local_discriminator.parameters():
        if param.grad is not None:
            param.grad.zero_()

    loss_sum =loss / len(trainloader.dataset)
    grads = torch.autograd.grad(loss_sum, list(local_discriminator.parameters()))
    grads = [grad_.cpu().numpy() for grad_ in grads]

    print(f"local Epoch {epoch+1}: Loss_local/-discriminator = {loss_sum:.4f}, for (Client {client_id})")

    return grads


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

#monn train and test

def init_net(output_dim, device="cpu"):
   
    n_classes=2
    net = ModelMOON(output_dim, n_classes)
    

    return net

def init_net(output_dim, device="cpu"):
   
    n_classes=2
    net = ModelMOON(output_dim, n_classes)
    

    return net

def train_moon(
    net,
    global_net,
    previous_net,
    train_dataloader,
    epochs,
    lr,
    mu,
    temperature,
    device="cpu",
    client_id=None
):
    """Training function for MOON."""
    net.to(device)
    global_net.to(device)
    previous_net.to(device)
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-5,
    )

    criterion = torch.nn.CrossEntropyLoss()
    

    previous_net.eval()
    for param in previous_net.parameters():
        param.requires_grad = False
    previous_net

    cnt = 0
    cos = torch.nn.CosineSimilarity(dim=-1)

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for _, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            if len(target.shape) == 1:
                target = target.unsqueeze(1)
            else:
                  target=target.squeeze(1)
            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            # pro1 is the representation by the current model (Line 14 of Algorithm 1)
            _, pro1, out = net(x)
            # pro2 is the representation by the global model (Line 15 of Algorithm 1)
            _, pro2, _ = global_net(x)
            # posi is the positive pair
            posi = cos(pro1, pro2)
            logits = posi.reshape(-1, 1)

            previous_net.to(device)
            # pro 3 is the representation by the previous model (Line 16 of Algorithm 1)
            _, pro3, _ = previous_net(x)
            # nega is the negative pair
            nega = cos(pro1, pro3)
            logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

            previous_net.to("cpu")
            logits /= temperature
            labels = torch.zeros(x.size(0)).long()
            # compute the model-contrastive loss (Line 17 of Algorithm 1)
            loss2 = mu * criterion(logits, labels)
            # compute the cross-entropy loss (Line 13 of Algorithm 1)
            loss1 = criterion(out, target)
            # compute the loss (Line 18 of Algorithm 1)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        print(
            "Epoch: %d Loss: %f Loss1: %f Loss2: %f"
            % (epoch, epoch_loss, epoch_loss1, epoch_loss2)
        )

    previous_net.to("cpu")
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)

    print(f">> Training accuracy: %f of client : {client_id}" % train_acc)
    net.to("cpu")
    global_net.to("cpu")
    print(" ** Training complete **")
    return net
    
def test_moon(net, test_dataloader, device="cpu"):
    """Test function."""
    net.to(device)
    test_acc, loss = compute_accuracy(net, test_dataloader, device=device)
    print(">> Test accuracy: %f" % test_acc)
    net.to("cpu")
    return test_acc, loss

def compute_accuracy(model, dataloader, device="cpu", multiloader=False):
    """Compute accuracy."""
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    
    criterion = torch.nn.CrossEntropyLoss()
   
    loss_collector = []
    if multiloader:
        for loader in dataloader:
            with torch.no_grad():
                for _, (x, target) in enumerate(loader):
                  
                    _, _, out = model(x)
                    if len(target) == 1:
                        loss = criterion(out, target)
                    else:
                        loss = criterion(out, target)
                    _, pred_label = torch.max(out.data, 1)
                    loss_collector.append(loss.item())
                    total += x.data.size()[0]
                    correct += (pred_label == target.data).sum().item()

                   
                    pred_labels_list = np.append(
                    pred_labels_list, pred_label.numpy()
                        )
                    true_labels_list = np.append(
                            true_labels_list, target.data.numpy()
                        )
                    '''
                    else:
                        pred_labels_list = np.append(
                            pred_labels_list, pred_label.cpu().numpy()
                        )
                        true_labels_list = np.append(
                            true_labels_list, target.data.cpu().numpy()
                        )
                    '''
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with torch.no_grad():
            for _, (x, target) in enumerate(dataloader):
                # print("x:",x)
                if len(target.shape) == 1:
                  target = target.unsqueeze(1)
                else:
                  target=target.squeeze(1)

                _, _, out = model(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

               
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
                '''
                else:
                    pred_labels_list = np.append(
                        pred_labels_list, pred_label.cpu().numpy()
                    )
                    true_labels_list = np.append(
                        true_labels_list, target.data.cpu().numpy()
                    )
                '''
            avg_loss = sum(loss_collector) / len(loss_collector)

    if was_training:
        model.train()

    return correct / float(total), avg_loss