# DEGCN
## What is it?
DEGCN,A dense graph convolutional network for multi-Omics data integration and kidney kancer subtype classification<br>
![Image text](https://github.com/yoyolooki/DEGCN/data/Figs1.png)
As shown in figure, inputs to the model are multi-omics expression matrices, including but not limited to genomics, transcriptomics, proteomics, etc. MoGCN exploits the GCN model to incorporate and extend two unsupervised multi-omics integration algorithms: Autoencoder algorithm (AE) based on expression matrix and similarity network fusion algorithm based on patient similarity network. Feature extraction is not necessary before AE and SNF. <br>



## Requirements 
MoGCN is a Python scirpt tool, Python environment need:<br>
Python 3.6 or above <br>
Pytorch 1.4.0 or above <br>
snfpy 0.2.2 <br>


```Python
#Kidney cancer experiment
python VAE_run.py -p1 data/KCdata/fpkm.csv -p2 data/KCdata/gistic.csv -p3 data/KCdata/rppa.csv -s 0 -d gpu -e 100 -m 0 -bs 16
python SNF.py -p KCdata/fpkm.csv KCdata/gistic.csv KCdata/rppa.csv -m sqeuclidean
python DenseGCN_run.py -fd result/latent_data.csv -ad result/SNF_fused_matrix.csv -ld data/KCdata/sample_classes.csv -ts KCdata/test_sample.csv -m 0 -d gpu -p 20

# Breast cancer experiment
# python VAE_run.py -p1 data/BCdata/fpkm.csv -p2 data/BCdata/gistic.csv -p3 data/BCdata/rppa.csv -s 0 -d gpu -e 100 -m 0 -bs 16
# python SNF.py -p BCdata/fpkm.csv BCdata/gistic.csv BCdata/rppa.csv -m sqeuclidean
# python DenseGCN_run.py -fd result/latent_data_bc.csv -ad result/SNF_fused_matrix_bc.csv -ld data/BCdata/sample_classes.csv -ts data/BCdata/test_sample.csv -m 0 -d gpu -p 20

# Gastric cancer experiment
# python VAE_run.py -p1 data/GCdata/CNV.csv -p2 data/GCdata/mRNA.csv -p3 data/GCdata/somatic.csv -s 0 -d gpu -e 100 -m 0 -bs 16
# python SNF.py -p data/GCdata/CNV.csv data/GCdata/mRNA.csv data/GCdata/somatic.csv -m sqeuclidean
# python DenseGCN_run.py -fd result/latent_data_gc.csv -ad result/SNF_fused_matrix_gc.csv -ld data/GCdata/sample_classes.csv -ts data/GCdata/test_sample.csv -m 0 -d gpu -p 20
```



