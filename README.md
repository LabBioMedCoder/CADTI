# CrossAttentionDTI
**CrossAttentionDTI: improving drug-target interaction prediction on heterogeneous networks by cross-attention**
<div align="center">
  <img src="https://github.com/LabBioMedCoder/CrossAttentionDTI/blob/main/CrossAttentionDTI.png" width="800px" height="300px">
</div>
Fig. 1. The framework of CrossAttentionDTI.The CrossAttentionDTI framework consists of three modules: A. Network feature extraction. B. Attribute feature extraction. C. Cross-attention feature fusion and prediction.

# Dependencies
* pytorch==1.12.0
* numpy==1.23.1
* pandas==1.4.3
* tensorboardX==2.6
* tensorflow==2.6.0
* keras==2.9.0
* scikit-learn==1.1.1
* RDKit==2022.9.3
* gensim==4.2.0
* subword-nmt==0.3.8

# Dataset
In the dataset folder, we provide the processed data of Luo_data.

# ESFP and dictionary
The ESFP folder contains the data needed for the FCS embedding method that is built based on MolTrans https://github.com/kexinhuang12345/MolTrans. The "dictionary" directory includes the dictionaries constructed for drugs and targets in the luo dataset.

# Resources
* dataset.py: data process.
* main.py: train and test the model
* hyperparameter.py: set the hyperparameter of CrossAttentionDTI
* model.py: CrossAttentionDTI model architecture
* measure.py: The module for calculating metrics

# Setting directory
Make the "result" directory before running the model. The run results are saved in the "result" directory

# Run
python main.py



