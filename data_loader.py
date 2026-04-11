from torch.utils.data import DataLoader, Dataset
import torch
import mne

#Machinery: DataLoader calls len(dataset) to get the total count of say 1000 trials. It generates a 
#list of indices [0, 1, 2, ..., 999]. It grabs the first batch size indices [0, 1, 2, 3] and calls
#dataset[0], dataset[1] ... (getitem here) and stacks all of those tensors into a batch tensor of
#shape (batch_size, channels, time_points) and sends that batch to the training loop. It should
#be noted that getitem just returns one tensor?

class EEGBCIDataset(Dataset):
    """
    Custom Dataset class for loading EEG data for BCI tasks. We need to build our own custom class to
    bridge the gap between the raw EEG data and the Pytorch DataLoader.
    """
    def __init__(self, subjects, runs):
        """
        Args:
            subjects (list): Subject Id's [1, 2, 3]
            runs (list): List of run identifiers [4, 8, 12]
        Initializes the dataset by loading the data for the specified subjects and runs.
        """
        self.epochs = []
        self.labels = []
        #Loop through each subject and load the data, standardise it, filter it for the bands we
        #want, extract events and cut into epochs. Finally store each trial and its label in the
        #dataset.
        for subject in subjects:
            file_names = mne.datasets.eegbci.load_data(subject, runs=runs)
            raws = [mne.io.read_raw_edf(f, preload=True) for f in file_names]
            raw = mne.concatenate_raws(raws)

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
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return X, y
    
class BandPowerDataset(Dataset):
    """
    Wraps EEGBCIDataset and extracts band power features (variance per channel)
    for the MLP. Standardizes features to mean 0, std 1 so the network can learn.
    Without standardization, EEG values in volts (1e-6) produce variances around 
    1e-10, which are too small for the network's default weight initialization (~0.1)
    to produce meaningful activations or gradients.
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