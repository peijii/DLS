# Domain-Generalization-by-Dynamic-Label-Smoothing-Strategy-for-Biosignals-Classification
Code for the model in the paper Domain Generalization by Dynamic Label Smoothing Strategy for Biosignals Classification
![overall structure](figure/framework.png)

# Datasets

>We evaluate the performance of our proposed dynamic label smoothing strategy on NinaPro DB1 datasets, which are open-access databases of Electromyography (EMG) recordings.

# Requirements

* Python 3.8
* Pytorch 1.11.0
* sklearn 0.24.0

# Function of file

* `a-layered-sensor-unit/main_experiment/model/ml/`
  * train machine learning model (XGBoost, SVM, RandomForest, KNN).
* `a-layered-sensor-unit/main_experiment/model/dl/model.py`
  * Generate sEMG-FMG LFN model, sEMG LFN model and FMG LFN model.

# Usage
We've offered three models:  `sEMG-FMG LayerFusionModel` , `sEMG LayerFusionModel` and `FMG LayerFusionModel` for dual modal (sEMG and FMG) and single modal (sEMG or FMG) respectively.
You need to use a tensor with shape: **[Batch_size, channel, length]** for all the three models.
