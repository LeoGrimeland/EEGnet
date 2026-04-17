import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from data_loader import EEGBCIDataset
from eegnet_model import EEGNet
from jepa_model import Predictor, mask_input, ema_update

#Hyperparams:
Batch_size=32
learning_rate = 1e-3
epochs = 50
tau = 0.996
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
subjects = list(range(1, 110))
runs = [4, 8, 12]

#Setup:

#Load all 109 subjects.
dataset = EEGBCIDataset(subjects, runs)
loader = DataLoader(dataset, batch_size=Batch_size, shuffle=True)

context_encoder = EEGNet().to(device)
target_encoder = EEGNet().to(device)
predictor = Predictor().to(device)

#Target weights starts identical to context weights.
target_encoder.load_state_dict(context_encoder.state_dict())

#Freeze target from autograd. Even if we forget with torch.nograd(), this prevents the target 
#encoders weights from being updated by the optimizer.
for param in target_encoder.parameters():
    param.requires_grad = False

#Optimizer. We want to optimize the context and predictor togheter as they are both being updated 
#by the loss, but not the target encoder as it is only being updated by the ema update.
optimizer = torch.optim.Adam(list(context_encoder.parameters()) + list(predictor.parameters()),
                              lr=learning_rate)

#Make a directory to save model checkpoints. 
os.makedirs('checkpoints', exist_ok=True)

#--TRAINING LOOP--

for epoch in range(epochs):
    #train simultaneuously, target stays in nograd.
    context_encoder.train()
    predictor.train()

    total_loss = 0
    for batch_idx, (X, y) in enumerate(loader):
        X = X.to(device)

        #Per trial normalization to match phase 4 and finetuning preprocessing.
        X = (X - X.mean(dim=(2,3), keepdim=True)) / X.std(dim=(2, 3), keepdim=True)

        #Mask the input.
        masked_x, positions = mask_input(X)

        #Context path with gradients.
        context_embeddings = context_encoder.encode(masked_x)

        #Target path without gradients.
        with torch.no_grad():
            target_encoder.eval()
            target_embeddings = target_encoder.encode(X)
        
        #Predict the masked out time points.
        predicted_embeddings = predictor(context_embeddings, positions)

        #Loss, MSE between prediction and target.
        loss = F.mse_loss(predicted_embeddings, target_embeddings)

        #Backpropogate and optimize.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #EMA update for target encoder. 
        ema_update(context_encoder, target_encoder, tau=tau)
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch + 1}/{epochs}: avg MSE loss: {avg_loss:.6f}")

    #Save embedding stats every 5 epochs. Eval and train here is important as we want want to avoid dropout
    #to measure our true embedding stats. We compute batch wise embedding variance, which is the mean
    #across the embedding dimensions of the per dimension variance. This will go to 0 if the encoder
    #outputs are constant. We also calculate the per dimension standard deviation range, (take the lo
    #west and higest value in the 320 length sdv vector)
    if epoch % 5 == 0:
        with torch.no_grad():
            context_encoder.eval()
            sample_batch, _ = next(iter(loader))
            sample_batch = sample_batch.to(device)
            #Normalize:
            sample_batch = (sample_batch - sample_batch.mean(dim=(2, 3), keepdim=True)) / sample_batch.std(dim=(2, 3), keepdim=True)
            embeddings = context_encoder.encode(sample_batch)
            mean_var = embeddings.var(dim=0).mean().item()
            per_dim_std = embeddings.std(dim=0)
            min_std = per_dim_std.min().item()
            max_std = per_dim_std.max().item()
            context_encoder.train()
            print(f"  embedding stats: mean_var={mean_var:.4f}, std range=[{min_std:.4f}, {max_std:.4f}]")
    
    #Checkpoint every 10 epochs.
    if (epoch + 1) % 10 == 0:
        torch.save(context_encoder.state_dict(), f'checkpoints/jepa_epoch_{epoch+1}.pt')

    if epoch == 0 and batch_idx == 0:
        print(f"context_emb: mean={context_embeddings.mean().item():.4f}, std={context_embeddings.std().item():.4f}")
        print(f"target_emb:  mean={target_embeddings.mean().item():.4f}, std={target_embeddings.std().item():.4f}")
        print(f"pred_emb:    mean={predicted_embeddings.mean().item():.4f}, std={predicted_embeddings.std().item():.4f}")
        print(f"loss: {loss.item():.6f}")
    
#Final save after training.
torch.save(context_encoder.state_dict(), 'checkpoints/jepa_final.pt')





