from torch.utils.data import DataLoader, Dataset
import torch
import mne
import numpy as np
import os

#Machinery: DataLoader calls len(dataset) to get the total count of say 1000 trials. It generates a 
#list of indices [0, 1, 2, ..., 999]. It grabs the first batch size indices [0, 1, 2, 3] and calls
#dataset[0], dataset[1] ... (getitem here) and stacks all of those tensors into a batch tensor of
#shape (batch_size, channels, time_points) and sends that batch to the training loop. It should
#be noted that getitem just returns one tensor.

class EEGBCIDataset(Dataset):
    """
    Custom Dataset class for loading EEG data for BCI tasks. We need to build our own custom class to
    bridge the gap between the raw EEG data and the Pytorch DataLoader.
    """
    def __init__(self, subjects, runs, cache_dir='./cache'):
        """
        Args: 
            subjects (list): Subject Id's [1, 2, 3]
            runs (list): List of run identifiers [4, 8, 12]
            cache_dir (str): Directory to save cached data
        Initializes the dataset by loading the data for the specified subjects and runs.
        """
        self.epochs = []
        self.labels = []
        self.groups = [] #For group k-fold cross validation.

        #Save data to a file.
        cache_file = os.path.join(cache_dir, f'eeg_s{subjects[0]}-{subjects[-1]}_r{"_".join(map(str, runs))}.npz')
        if os.path.exists(cache_file):
            print(f'Loading cached data from {cache_file}')
            cached = np.load(cache_file)
            self.epochs = list(cached['epochs'])
            self.labels = list(cached['labels'])
            self.groups = cached['groups']
            self.labels_array = cached['labels_array']
            print(f'Loaded {len(self.labels)} trials from cache')
            return

        #Loop through each subject and load the data, standardise it, filter it for the bands we
        #want, extract events and cut into epochs. Finally store each trial and its label in the
        #dataset.
        for subject in subjects:
            n_before = len(self.labels) #count trials before loading this subject.

            file_names = mne.datasets.eegbci.load_data(subject, runs=runs)
            raws = [mne.io.read_raw_edf(f, preload=True) for f in file_names]
            raw = mne.concatenate_raws(raws)
            raw.resample(160) #Resample so every trial has the expected number of time points. 

            #Clean channel names and set a montage (as in notebook).
            mne.datasets.eegbci.standardize(raw)
            montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(montage)

            #Bandpass filter to mu and beta bands (8-30 Hz).
            raw.filter(l_freq=8, h_freq=30)

            #Extract the events and map to our two classes (right hand, left hand).
            events, event_id = mne.events_from_annotations(raw)
            #T1 = left hand, T2 = right hand (for runs 4, 8, 12).
            event_id_map = {'left': event_id['T1'], 'right': event_id['T2']}

            #Cut the continous data into epochs (trials) of 4 seconds each.
            trial_epochs = mne.Epochs(raw=raw, events=events, event_id=event_id_map, tmin=0, tmax=4, 
                                      baseline=None, preload=True)

            #Get the numpy arrays for the epochs and labels. Shape: (n_trials, n_channels=64,
            #n_time_points=641)
            data = trial_epochs.get_data()
            #labels as 0 = left, 1 = right.
            label_array = trial_epochs.events[:, 2]
            labels = [0 if label == event_id_map['left'] else 1 for label in label_array]

            #Store each trial individually in the dataset.
            for i in range(len(data)):
                self.epochs.append(data[i])
                self.labels.append(labels[i])
                self.groups.append(subject) #Keep track of which subject has which trials.
            
            n_after = len(self.labels) #Count trials after loading this subject.
            print(f"Subject {subject}: {n_after - n_before} trials loaded")

            #explicitly delete MNE objects to free RAM before next subject
            del raw, raws, trial_epochs, data

        #Convert lists to numpy arrays for efficientindexing by sklearn. GroupKfolds expects arrays.
        self.groups = np.array(self.groups)
        self.labels_array = np.array(self.labels)

        #Save to cache for next time.
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache_file,
                 epochs=np.array(self.epochs),
                 labels=self.labels_array,
                 groups=self.groups,
                 labels_array=self.labels_array)
        print(f'Cached {len(self.labels)} trials to {cache_file}')

    def __len__(self):
        """
        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            return one (tensor, label) pair, where tensor is the EEG data and label is the corresponding class label.
        """
        X = torch.tensor(self.epochs[idx], dtype=torch.float32)
        X = X.unsqueeze(0) #Add a channel dimension for CNN input, shape becomes (1, 64, 641).
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return X, y
    
class BandPowerDataset(Dataset):
    """
    Wraps EEGBCIDataset and extracts band power features (variance per channel)
    for the MLP. Standardizes features to mean 0, std 1 so the network can learn.
    Without standardization, EEG values in volts (1e-6) produce variances around 
    1e-10, which are too small for the network's default weight initialization (~0.1)
    to produce meaningful activations or gradients. Gradients would be near zero 
    and the network would not learn.
    """
    def __init__(self, raw_dataset):
        all_features = []
        all_labels = []
        for i in range(len(raw_dataset)):
            X, y = raw_dataset[i]
            features = X.var(dim=-1)
            # X is shape (64, 641). Variance across time gives band power per channel.
            all_features.append(features)
            all_labels.append(y)
        #Stack em into tensors for easier indexing during training.
        self.features = torch.stack(all_features) #(n_trials, 64)
        self.labels = torch.stack(all_labels) #(n_trials,)

        #Compute per channel mean and std across all trials for standardization.
        self.mean = self.features.mean(dim=0) #(64,)
        self.std = self.features.std(dim=0) #(64,)

        #Standardize features to mean 0, std 1.
        self.features = (self.features - self.mean) / self.std

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]