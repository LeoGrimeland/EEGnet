import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset
from data_loader import EEGBCIDataset
from eegnet_model import EEGNet

#Finetuning for JEPA pretrained model. Copied the same train file from train.py, but will use the
#pretrained weights from the google drive to which they were saved.

PRETRAINED_PATH = '/content/drive/MyDrive/bci_project/checkpoints/jepa_final.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

#Hyperparameters:
learning_rate = 1e-4
batch_size = 64
subjects = list(range(1, 110))
runs = [4, 8, 12]
n_folds = 5
n_epochs = 30 #More data means faster convergance, we work with fewer epochs. 

dataset = EEGBCIDataset(subjects, runs)

#Extract all features and labels into arrays for splitting. 
all_X = []
all_y = []
for i in range(len(dataset)):
    X, y = dataset[i]
    all_X.append(X)
    all_y.append(y)

features = torch.stack(all_X)
labels = torch.stack(all_y)
groups = dataset.groups

print(f'Total trials: {len(features)}')
print(f'Feature shape: {features.shape}')
print(f"Unique subjects: {len(set(groups))}")

# Standardize: compute mean and std across all trials, per channel, per timepoint.
# For CNNs on raw signal, a simpler approach works — normalize per trial to mean 0, std 1.
# This ensures each trial has comparable scale regardless of amplitude drift between runs.
mean = features.mean(dim=(2, 3), keepdim=True)  # mean across trials and the 1-channel dim
std = features.std(dim=(2, 3), keepdim=True)
features = (features - mean) / std

#--K-folds cross validation loop.--

#These lines set set up the k folds splitting, using stratified splitting to ensure equal 
#distibutions of the two classes in each fold. fold_accuracies holds the score for each fold
#which we will later average. random_state makes the folds reproducible.
kfold = GroupKFold(n_splits=n_folds)
fold_accuracies = []

#For loop iterates over each fold, (kfold.split returns the indices for the training and 
#testing sets for that fold).)
for fold, (train_indices, test_indices) in enumerate(kfold.split(features, labels, groups)):
    #Split the data into training and testing sets. We index into features, so if
    #features is shape (45, 64) then X_train will be shape (36, 64) (80% of 45 trials).
    X_train, X_test = features[train_indices], features[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]

    #Report which subjects are in the test set for this fold.
    test_subjects = sorted(set(groups[test_indices]))
    print(f'\nFold {fold+1}: testing on {len(test_subjects)} subjects: {test_subjects[:5]}...')

    #Wrap training data in dataloader for batching and shuffling.
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    #Create a fresh model for each fold so it does not carry over anything from previous folds.
    model = EEGNet().to(device)
    pretrained_state = torch.load(PRETRAINED_PATH, map_location=device)
    missing, unexpected = model.load_state_dict(pretrained_state, strict=False)
    if fold == 0:
        print(f"  Loaded pretrained weights. Missing: {missing}, Unexpected: {unexpected}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #Training loop for the current fold. Epochs here is the particular training run, not the 
    #time windows we split our EEG data into.
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.00
        #Shuffle and batch the training data for this epoch.
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            #Calls forward method. If X_batch is shape (8, 64) then predictions will be shape
            #(8, 2) for 8 samples and 2 output classes. Each value is the logit (raw score) for 
            #that class.
            predictions = model(X_batch)
            #Internally softmaxes logits to probablity scores. 
            loss = criterion(predictions, y_batch)
            optimizer.zero_grad()
            #Computes gradients with respect to the loss for each parameter.
            loss.backward()
            #We use adam, updates the weights based on the computed gradients to minimise the loss.
            optimizer.step()

            #Max-norm constraint on depthwise convolution weghts to prevent them from growing too
            #large and overfitting. We do this after the weights have been updated by the optimizer.
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'depthwise_conv' in name:
                        # param shape is (F1*D, 1, n_channels, 1)
                        # Compute norm per filter (dim 0), across the spatial weights (dim 2)
                        norms = param.data.norm(2, dim=2, keepdim=True)
                        # Clamp: if norm > 1, scale down. If norm <= 1, leave unchanged.
                        desired = param.data * (1.0 / norms.clamp(min=1.0))
                        param.data.copy_(desired)
            #Accumulate the loss for this epoch to track progress. Print every 10 epochs.
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f'Fold {fold+1}, Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(train_loader):.4f}')
    
    model.eval()
    with torch.no_grad():
        test_X = X_test.to(device)
        test_y = y_test.to(device)
        outputs = model(test_X)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == test_y).sum().item()
        accuracy = correct / len(test_y)
        fold_accuracies.append(accuracy)
        print(f'Fold {fold+1} accuracy: {accuracy:.4f} ({correct}/{len(test_y)})')


print(f'\nMean accuracy: {sum(fold_accuracies)/len(fold_accuracies):.4f}')
print(f'Per-fold: {fold_accuracies}')
















