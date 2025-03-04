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
1.Use VAE to reduce the dimensionality of multi-omics data to obtain multi-omics feature matrix
```Python
python VAE_run.py -p1 data/KCdata/fpkm.csv -p2 data/KCdata/gistic.csv -p3 data/KCdata/rppa.csv -s 0 -d gpu -e 100 -m 0 -bs 16
```
2.Use SNF to construct patient similarity network <br>
```Python
python SNF.py -p KCdata/fpkm.csv KCdata/gistic.csv KCdata/rppa.csv -m sqeuclidean
```
3.Input multi-omics feature matrix  and the patient similarity network to GCN <br>
```Python
python DenseGCN_run.py -fd result/latent_data.csv -ad result/SNF_fused_matrix.csv -ld data/KCdata/sample_classes.csv -ts KCdata/test_sample.csv -m 0 -d gpu -p 20
```

## Contact
For any questions please contact Yu Li (Email: 2023215215005@stu.ahtcm.edu.cn).

## Citation

