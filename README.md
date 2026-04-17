Project EEGNet. 

Phase 0:

---Rewrite---

Objective
Explore the PhysioNet EEG Motor Movement/Imagery Dataset for one subject to understand the structure of EEG data and identify the neural signal (Event-Related Desynchronization) that underpins motor imagery classification.
Dataset

Source: PhysioNet EEG Motor Movement/Imagery Dataset (Subject 1)
Recording setup: 64-channel EEG, 10-20 electrode system, 160 Hz sampling rate
Task: Left fist vs right fist motor imagery (runs 4, 8, 12)
Trials: ~45 imagery trials per condition across 3 combined runs

What Was Done

Data loading and inspection — Downloaded EDF files via MNE-Python, inspected the Raw object structure (64 channels × ~20,000 samples per run), and verified channel names and sampling rate.
Electrode visualization — Plotted the 10-20 sensor layout and identified C3 (left motor cortex) and C4 (right motor cortex) as the key channels for left vs right hand imagery.
Raw signal inspection — Plotted raw EEG time series from C3 and C4. Confirmed that motor imagery events are not visible in raw waveforms — the signal lives in the frequency domain.
Event extraction — Extracted event annotations (T0=rest, T1=left imagery, T2=right imagery) and visualized the experimental timeline showing alternating trial structure.
Power spectral density — Computed PSD using Welch's method at C3 and C4. Observed the 1/f power falloff and identified the mu rhythm peak at 8–13 Hz.
Condition comparison (PSD) — Compared mu-band power between left and right imagery conditions at C3 and C4. Differences were noisy with limited trials.
Time-frequency analysis — Used Morlet wavelet decomposition to visualize power changes over time at C3. Observed some ERD patterns but noisy due to limited trial averaging.
Topographic mapping — Computed mu-band power difference (right − left imagery) across all 64 channels and plotted as a scalp topographic map. Result showed clear lateralized ERD: blue (suppression) over left motor cortex during right imagery, red over right motor cortex during left imagery.

Key Findings

ERD is real and lateralized in this subject. The topographic map shows a clean left-right split in mu-band power difference, confirming that motor imagery produces detectable, spatially specific changes in neural activity.
Single-trial EEG is noisy. Time-frequency plots from individual runs were difficult to interpret. Averaging across more trials and using spatial analysis (topomaps) revealed the pattern more clearly.
The classification signal is the difference in mu/beta band power (8–30 Hz) over contralateral motor cortex — C3 for right hand imagery, C4 for left hand imagery. 

Phase 2:

Built train and model for a simple MLP. First run achieved accuracy of 48% which is worse than a 
guess. Built an additional data loader class to handle the type of data a MLP needs as input, here the band power across 64 channels (8, 64). Had to get the logic for computing band power from train
to data_loader, and adding standardisation improved performance to 62%. Strong indications of overfitting, however changing the hidden layer size up or down from 32 decreased performance to 52-56%. 

Phase 3:

For phase 2 with the MLP we took each trials epoch of shape (64 channels, 641 time points) and collapsed the time dimension to get a flat vector of shape (64,) where each entry was a power approximation (using variance of the whole bands oscillations). THis was fed into an MLP, 64 features per trial. However, we dscarded all temporal structure, when power changedwhich is key to ERD. We also locked ourselves into one specific band as we had to bandpass in order to approprietely process the data. It is better if the data tells us what matters. 

We need a better approximation of motor activity classification, we need to distinguish how power drops at specific frequencies at specific times over spatially related electrodes. For phase 3 we create a CNN which we feed the raw data of shape (64, 641), no more collapsed time dimension, basically a snapshot of brain activity across space and time for one trial. 

Pipeline: Raw EEG → bandpass filter → epoch → (64, 641) tensor → EEGNet → classify

To do this we create a CNN, which slides a filter of weights across an input to detect certain patterns. This makes features recognizable over time (as a detected feature at time t will be detected by the same filter at time t+100) and reduces the number of parameters (as one filter covers a entire dimension rather than learning a seperate weight for every input position) which is critical for low training data like we see here. For EEG specifically, this means sliding a temporal filter across the input signal to detect mu band oscillations (10 Hz) wherever they appear. We computed mu band power manually for the MLP, here the model will come to learn power band features itself from the input data. 

This temporal convolution fitler is 1D, and is essentially a bandpass. We then use a seperate filter (depht wise spatial convolution) to slide across the signal axis. This learns where on the scalp each frequency pattern is important (C3 and C4 maybe Cz for mu and beta rythms). This is the equivalent of the topographic map of the electrode positions on the scalp and relative band power related to that position (computed by hand in notebook). We then use a second temporal filter that captures how the filtered, spatially weighted signal evolves over time (capturing when ERD occurs). THis can be related to a learned spectogram with time on the x axis, frequency on the y axis and power represented by brightness. Since we are no longer computing power by variance proxy manually, we will use our EEGBCIDataset as opposed to our BandPowerDataset. Caveat: input is actually (1, 64, 641), the 1 is the channel dimension here. 

Achieved poor performance on the first run, mean accuracy at 0.4889. This is most likely because we only have one subject. Changed dropout from 0.5 to 0.25 and performance improved to 0.6, however the
training loss was very small at the end of each fold (near 0), which is a major indication of overfitting. With an MLP, working with a small dataset of 45 trials was fine because we injected domain knowledge directly, but this is harder with a CNN because it needs to learn these structures itself. 

Phase 4:

Load in the data from all 109 subjects. This will help with training the CNN model but presents other challenges. We have to change our K-folds strategy to instead cycle through 109 subjects, and ensure that no test data is accidentaly leaked to the training data. This is a challenge, making a model that can generalise across subjects. EEG data is very different for different people depening on measurement noise, and ERD strength varies between people. 

We will modify the data_loader to handle this. We also need to ensure that sklearn can track which trials belong to whch subject. To do this we add a groups list, which tracks what trials belong to which individual. We also modify train.py to prevent data leakage. If K-folds splits subject 7 into 36 train trials and 9 test trials it does not need to recognise motor patterns in order to have good performance, it can just memorise subject 7's brain signature and recall what labels went with that signature. We need it to learn the universal ERD pattern. Sklearn provides GroupKFold, which works exactly like StratifiedKFold except it respects group boundaries. You pass it a groups array, and it guarantees that all trials from the same group (subject) stay in the same fold. If Subject 7 is in the test fold, all 45 of Subject 7's trials are in the test set, and zero are in training.

Bug when trying to run the model on 109 subject: trial 3899 has shape (1, 64, 513) meaning it was not sampled at the same rate as others (160 Hz). Modified data_loader to resample every trial to 160.

With all of the available data (109 subjects) we achieved a performance of 0.7514, higher than some papers out there. 

Phase 5:

Noah Frontend.

Phase 6: 

Jepa model. Structure is going to deviate from industry standard here. We need a context model to produce the 320 length vector of masked embeddings (masked parts is what the predictor model will try to recreate). This will be of the same EEGNet type as in eegnet_model.py, altough with a detached final linear layer, so its output will be the pure 320 length vector of feature maps (with some masked) produced by the final convolutional layer in the context encoder. This vector is flattened, so it can be used as a input to the midlevel MLP (made up of linear layers) along with a 2D vector of the indices of the masked portion of the feature maps. This midlevel layer is the predictor, whose function is to estimate the values of the masked portion of the input vector. It compares its output to the output of the target encoder, which again is a EEGNet whose output is the full representation of the same signal passed through the context encoding model. The error between that prediction and the actual values produced by the target encoder will be measured and minimised using L2 norm (MSE). 

The industry standard is to use transformers for the context and target encoders, as well as the prediction layer. The transformer arhictecture maps naturally to the task of masking a part of an image or signal due to its ability to break the input into chunks and learn about how these patches infuence each other (easy to mask patches of input). We will deviate from this standard here as the logic for EEGNets is already established and customised to our EEG data, we are less familiar with the transformer architecture and most importantly transformers are very data hungry. We have a limited and rather small dataset (64 channels x 641 time points, 41,000 values per trial, and we have 4900 trials across 109 subjects). Transformers must learn the relationship between features through data, that electrodes that are spatial neighbours tend to coactivate. EEGNet's architecture gives us this for free, as it is specially built for EEG data (temporal filtering -> spatial filtering -> temporal pattern detection) and convolutional kernels inherently capture local patterns. These features of EEGNet's allows us to implement a Jepa architecture with limited data, transformers would have to rediscover what the EEGNet's architecture already captures. Thus: We use EEGNet as the encoder backbone rather than a Vision Transformer, as is standard in I-JEPA, because the inductive biases of the convolutional architecture are better matched to our data regime (4,900 trials). Transformer based encoders would be a natural extension given a larger EEG corpus

The biggest problem that arises with the use of EEGNet encoders is with the masking of a portion of the input. While a transformer can simply mask a patch of input as removing patches does not break the model, our EEGNet expects a full (1, 64, 641) input tensor to which our kernel size is calibrated. To solve this we substitute patch masking with simply zeroing out a region of the input. The context encoder still sees the full tensor, but our masked portion has all zeros. It should be mentioned that this is a tiny waste of computation, as the zeros still have to be processed. So what are we masking? Here we will mask the temporal signal of the EEG, as the features we want the modle to recognise generally evolve over time in the form of a ERD. We could also implement spatial masking across electrodes, which is a reasonable future extension.

For our loss function we will use L2 MSE, specifically loss = MSE(predictor(context_output, masked_indices), target_output.detach()). The detach is to stop gradient flow through the target encoder. This is done for a few deep reasons:

The collapse problem. 

Here is the pipeline:

Step 1 — Modify EEGNet to be a headless encoder. Factor out the classifier head so you have a clean encode() method that returns the 320-dim embedding. The classifier becomes a separate module you attach for fine-tuning.

Step 2 — Build the masking module. A function that takes an EEG tensor (batch, 1, 64, 641), picks a random contiguous time block to mask (say, 25–50% of the timepoints), zeros it out, and returns both the masked tensor and the mask position info.

Step 3 — Build the predictor MLP. Something like (320 + 2) → 256 → 256 → 320. Small and deliberate — remember, too much capacity here lets the predictor cheat.

Step 4 — Build the EMA update function. After each optimizer step, update target encoder weights: θ_target = τ * θ_target + (1 - τ) * θ_context.

Step 5 — Write the pretraining loop. For each batch: mask the input, pass masked input through context encoder, pass full input through target encoder (with torch.no_grad()), run predictor, compute MSE loss on normalized embeddings, backprop through predictor and context encoder only, then do EMA update.

Step 6 — Write the fine-tuning script. Load the pretrained context encoder weights, attach a fresh classifier head, freeze the encoder (or use a low learning rate on it), train on the labeled left-vs-right task. Compare accuracy to your Phase 4 baseline.

Step 7 — Ablation and analysis. Run without pretraining as control. Try different mask ratios. Try freezing vs. unfreezing the encoder during fine-tuning. This is what makes the portfolio piece convincing — it shows you understand the design space, not just one configuration.