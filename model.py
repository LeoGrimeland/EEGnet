import torch
import torch.nn as nn

#Builds simple MLP model. This will be used in phase 2 to verify the data loading and 
#training pipeline before we move on to the more complex CNN model. To prevent overfitting
#on the small dataset, we will use only one hidden layer with a small number of neurons. Input 
#is size 64 (band power features for each channel) and output is size 2 (left vs right hand movement).

class EEGMLP(nn.Module):
    def __init__(self, input_size=64, hidden_size=24, output_size=2):
        #super() used to call the __init__() method of the parent class.
        super().__init__()
        #Define layers.
        self.fc1 = nn.Linear(input_size, hidden_size)
        #We use ReLU, which introduces non-linearity to the model by zeroing out negative values.
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #Pass the input through the layers and activations.
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


