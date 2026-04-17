import torch
import torch.nn as nn
import numpy as np

def mask_input(x):
    """
    Masks the input to the context encoder by randomly masking out a contigous segment of time points.
    """
    #x of shape (Batch, 1, 64, 641).
    B, _, C, T = x.shape
    masked_x = x.clone()
    positions_list = []

    #For each sample in the batch, randomly sample a mask ration in range [0.25, 0.50].
    for sample in range(B):
        mask_ratio = np.random.uniform(0.25, 0.50)

        #Convert mask ratios into mask lengths (int number of time points to mask out).
        mask_length =  int(mask_ratio * T)

        #For each sample, randomly sample a valid start index in range [0, T - mask_length] so
        #[start + mask_length <= T].
        start_index = np.random.randint(0, T - mask_length + 1)

        #Build the masked out tensor by cloning x, then zeroing out the masked time points for
        #each sample. 
        masked_x[sample, :, :, start_index : start_index + mask_length] = 0

        #Add position indeces of start and end to list and normalize.
        positions_list.append([start_index/T, (start_index + mask_length)/T])

    #Build the positions tensor which tell the context encoder which time points were
    #masked. Normalize the positions to be in range [0, 1] by dividing the time point
    #indices by T. Stack for each sample.
    positions = torch.tensor(positions_list, dtype=torch.float32, device = x.device)

    return masked_x, positions
    
class Predictor(nn.Module):
    """
    Class containig archutecture and forward method of the midlevel MLP predictor which takes
    the output of the context encoder and predicts the masked out time points. 
    """
    def __init__(self, embedding_dim=320, position_dim=2, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim + position_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, embedding, positions):
        """
        Forward pass for the predictor. Output is predicted embeddings.
        """
        #Concatenate the embedding and positions along the feature dimension to get a combined 
        #input of shape (batch, embedding_dim + position_dim). dim=1 because we want to 
        #concatenate along the feature dimension, not the batch dimension.
        combined_input = torch.cat((embedding, positions), dim=1)

        #Send through MLP to get the predicted embedding for the masked out time points.
        predicted_embedding = self.mlp(combined_input)

        #Return predicted embedding of shape (batch, 320).
        return predicted_embedding
    
#This is the main weight uodae logic for the target encoder. We use torch.nograd() to ensure that
#gradients are not flowing through the target encoders weights during training, as this would lead
#to representation collapse. 
@torch.no_grad()
def ema_update(context_encoder, target_encoder, tau=0.996):
    """
    Updates the weights of the target encoder using an exponential moving average of the 
    conext encoder weights. Tau=0.996 creates meaningful lag for the target encoder (250 steps).
    """
    for context_param, target_param in zip(
        context_encoder.parameters(), target_encoder.parameters()
    ):
        #target ← τ·target + (1-τ)·context
        target_param.data = tau * target_param.data + ((1- tau) * context_param.data)
    
    #Also update the buffers for batchnorm layers, which are not parameters but also need to be
    #updated with EMA. (BN running mean/var). 
    for context_buf, target_buf in zip(
        context_encoder.buffers(), target_encoder.buffers()
    ):
        target_buf.data.copy_(context_buf.data)


    




    
