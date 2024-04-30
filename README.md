# Dynamic Label Smoothing Strategy for Biosignals Classification
Code for the model in the paper Dynamic Label Smoothing Strategy for Biosignals Classification. The detailed code now available [here](https://github.com/peijii/I2CNet).
![overall structure](figure/framework.png)

# Datasets

>We evaluate the performance of our proposed dynamic label smoothing strategy on NinaPro DB1 datasets, which are open-access databases of Electromyography (EMG) recordings. Information on how to obtain it can be found [here](https://ninapro.hevs.ch/instructions/DB1.html).

# Requirements

* Python 3.8
* Pytorch 1.11.0
* sklearn 0.24.0

# Function of file

* `DLS/models/feature_extractor.py`
  * feature extractor: The first component of our framework is responsible for extracting features from multi-channel time series data. Its architecture and inner workings will be detailed in an upcoming paper. The Feature Extractor plays a crucial role in transforming raw time series data into a more compact and informative representation that can be used by the following components.
* `DLS/models/label_predictor.py`
  * label predictor: The second component, the Label Predictor, is analogous to the classification layer found in traditional feedforward neural networks. It takes the features extracted by the Feature Extractor and performs classification tasks. The Label Predictor utilizes various activation functions and optimization techniques to map the extracted features to corresponding class labels.
* `DLS/models/label_adjustor.py`
  * label adjustor: The final component of our framework is the Label Adjustor, which utilizes the features extracted by the Feature Extractor to adjust the true labels. The purpose of this component is to enable the Feature Extractor to learn both inter-and inter- samples differences, thus improving the model's generalization ability. In cases where the input data contains noise, the Label Adjustor can help maintain or even enhance the model's performance by refining the true label based on t
