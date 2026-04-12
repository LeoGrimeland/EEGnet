import torch
import torch.nn as nn

#Step 1: first temporal filter. Say we have a input of a 1D vector of size 641, we slide a filter 
#of say 32 weights across the time dimension to produce a single output value, (one for 0-32, 
#one for 1-33 etc...). This value is high if the underlying signal in that time window matches the
#pattern the filter is looking for, and this pattern is detected anywhere. This is how CNN's learn
#structure. We will repeat this for each of the 64 channels, our input is 2D so we use conv2D to
#process the whole time dimension for each filter but one channel at a time. (Filter shape: (1, 1, 32))
#We use F1 filters, so the output will be (F1, 64, 641) where each of the F1 feature maps is the 
#entire 64 channel EEG, filtered at one learned frequency band.

#Step 2: depthwise convolution. We apply spatial filters to the feature map from step 1 to
#learn which electrodes matter for which frequency patterns. We use depthwise here as we
#want to learn seperate spatial filters for each frequency pattern, we dont want to mix
#information across feature maps.

#Step 3: We split the last convolution into a depthwise and pointwise convolution to reduce the
#number of parameters and prevent overfitting. The depthwise convolution learns more complex temporal
#patterns across the whole feature map, and the pointwise convolution learns how to combine the
#output of the depthwise convolution across feature maps to learn more complex patterns that may 
#involve multiple frequency bands.

class EEGNet(nn.Module):
    #F1=number of filters to learn, kernel_length=length of filter, 160 Hz, 32 samples = 200 ms
    #time window.
    def __init__(self, F1=8, F2=16, kernel_length=32, n_channels=64, n_classes=2, D=2):
        super().__init__()

        #Padding to include all time points (filter size 32 leaves some time points out).
        #Bias is set to false as batch normalization handles this later.
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding='same', bias=False)

        #Batch normalization helps with training stability by preventing mean and variance drift
        #as weights are updated. It normalizes the output of the convolutional layer to have mean 0 and
        #std 1 per feature map, which allows the network to learn more effectively.
        self.batchnorm1 = nn.BatchNorm2d(F1) #Output (batch, F1, 64, 641)

        #We want convolution with shape (64, 1) to span all 64 electrodes at 1 timepoint, to 
        #compute the weighted sum across al channels at each time point for each frequency pattern.
        #The weights are a spatial filter, which electrodes matter. Groups=F1 ensures each filter 
        #sees only 1 input feature map.Depth multiplier D is the number of spatial filters we learn
        #per temporal filter, so output is F1*D feature maps (some frequency patterns may have
        #multiple informative spatial patterns, onset, rebound etc...). 
        self.depthwise_conv = nn.Conv2d(F1, F1*D, (n_channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1*D)

        #We use ELU here as it allows for negative values to be passed to the next layer,
        #preventing "dead neurons" which one gets with ReLu. ELU: if X<0 output=exp(x)-1.
        self.elu = nn.ELU()

        #Average pooling to reduce dimensionality and make the network more robust to small shifts in time.
        self.avgpool = nn.AvgPool2d((1, 4)) #Every 4 time points get averaged togheter to 1 value.

        #Drpout to reduce reliance on any one feature and prevent overfitting.
        self.dropout = nn.Dropout(0.5) #Output: (batch, D*F1, 1, 160)

        #Depthwise temporal cnonv2D again to learn more complex temporal patterns across the 
        #whole feature map. 
        self.separable_conv_depth = nn.Conv2d(F1*D, F1*D, (1, 16), padding='same', groups=F1*D, bias=False)
        self.separable_conv_point = nn.Conv2d(F1*D, F2, (1, 1), groups=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, 8)) #Output (batch, F2, 1, 20)

        #Pass this output through a fully connected nn.Linear layer to get the final 2 logit outputs.
        #First we flatten the (batch, F2, 1, 20) output to 1D vector of length 320 features.
        self.classifier = nn.Linear(F2 * (641 // 4 // 8), n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise_conv(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.separable_conv_depth(x)
        x = self.separable_conv_point(x)
        x = self.batchnorm3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout(x)

        #Flatten the output to (batch, F2*20) for the classifier.
        x = x.view(x.size(0), -1)

        logits = self.classifier(x)
        return logits











