import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from data_loader import EEGBCIDataset, BandPowerDataset
from model import EEGMLP

#For training our model we can not use the customary 80/20 split, as the dataset is very small
#and accuracy might drop inordinately if 1 trial was misclassified (11% drop). Instead we will use
#k-folds cross validation. Here we will use 5 folds, meaning that for our 45 trials (per run per 
#subject) we will train on 36 trials and test on 9 trials, and then repeat this process 5 times
#so that each trial is used for testing once and training multiple times. We then average over the 5
#accuracy scores to get a more robust estimate of the models performance. 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

#Hyperparameters:
learning_rate = 0.001
batch_size = 8
subjects = [1]
runs = [4, 8, 12]
n_folds = 5
n_epochs = 50

raw_dataset = EEGBCIDataset(subjects, runs)
dataset = BandPowerDataset(raw_dataset)

#Extract all features and labels into arrays for splitting. 
features = dataset.features
labels = dataset.labels

print(f'Total trials: {len(features)}')
print(f'Feature shape: {features.shape}')

#--K-folds cross validation loop.--

#These lines set set up the k folds splitting, using stratified splitting to ensure equal 
#distibutions of the two classes in each fold. fold_accuracies holds the score for each fold
#which we will later average. random_state makes the folds reproducible.
kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
fold_accuracies = []

#For loop iterates over each fold, (kfold.split returns the indices for the training and 
#testing sets for that fold).)
for fold, (train_indices, test_indices) in enumerate(kfold.split(features, labels)):
    #Split the data into training and testing sets. We index into features, so if
    #features is shape (45, 64) then X_train will be shape (36, 64) (80% of 45 trials).
    X_train, X_test = features[train_indices], features[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]

    #Wrap training data in dataloader for batching and shuffling.
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    #Create a fresh model for each fold so it does not carry over anything from previous folds.
    model = EEGMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #Training loop for the current fold. Epochs here is the particular training run, not the 
    #time windows we split our EEG data into.
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.00
        #Shuffle and batch the training data for this epoch.
        for X_batch, y_batch in train_loader:
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
            #Accumulate the loss for this epoch to track progress. Print every 10 epochs.
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss / len(train_loader):.4f}')
        
        #Evaluate on test fold.
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        predicted_classes = test_predictions.argmax(dim=1)
        correct = (predicted_classes == y_test).sum().item()
        accuracy = correct / len(y_test)
        fold_accuracies.append(accuracy)
        print(f'  Fold {fold + 1} accuracy: {accuracy:.2f} ({correct}/{len(y_test)})')

mean_acc = sum(fold_accuracies) / len(fold_accuracies)
print(f'\nMean accuracy across {n_folds} folds: {mean_acc:.2f}')
















