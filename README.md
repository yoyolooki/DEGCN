# DEGCN
## What is it?
DEGCN,A dense graph convolutional network for multi-Omics data integration and kidney kancer subtype classification<br>
![Image text](https://github.com/yoyolooki/DEGCN/blob/main/data/Figs1.png)
As shown in figure, inputs to the model are multi-omics expression matrices, including but not limited to genomics, transcriptomics, proteomics, etc. DEGCN exploits the GCN model to incorporate and extend two unsupervised multi-omics integration algorithms: Variational Autoencoder algorithm (VAE) based on expression matrix and similarity network fusion algorithm based on patient similarity network. Feature extraction is not necessary before VAE and SNF. <br>

## Getting Started
### Step 1 Installation
Set up conda environment and clone the github repo.
```
# create a new environment
$ conda create --name DEGCN
$ conda activate DEGCN
# install requirements
$ pip install -r requirements.txt
```
###  Step 2 Running
The whole workflow is divided into three steps: <br>
* Use VAE to reduce the dimensionality of multi-omics data to obtain multi-omics feature matrix
```Python
python VAE_run.py -p1 data/KCdata/fpkm.csv -p2 data/KCdata/gistic.csv -p3 data/KCdata/rppa.csv -s 0 -d gpu -e 100 -m 0 -bs 16
```
* Use SNF to construct patient similarity network <br>
```Python
python SNF.py -p KCdata/fpkm.csv KCdata/gistic.csv KCdata/rppa.csv -m sqeuclidean
```
* Input multi-omics feature matrix  and the patient similarity network to GCN <br>
```Python
python DenseGCN_run.py -fd result/latent_data.csv -ad result/SNF_fused_matrix.csv -ld data/KCdata/sample_classes.csv -ts KCdata/test_sample.csv -m 0 -d gpu -p 20
```

### Data Format
* The input type of each omics data must be .csv, the rows represent samples, and the columns represent features (genes). In each expression matrix, the first column must be the samples, and the remaining columns are features. Samples in all omics data must be consistent. AE and SNF are unsupervised models and do not require sample labels.<br>
* GCN is a semi-supervised classification model, it requires sample label files (.csv format) during training. The first column of the label file is the sample name, the second column is the digitized sample label, the remaining columns are not necessary. <br>

## Contact
For any questions please contact Yu Li (Email: 2023215215005@stu.ahtcm.edu.cn).

## Citation

