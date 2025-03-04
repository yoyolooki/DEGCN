# DEGCN

## Overview
**DEGCN** (Dense Enhanced Graph Convolutional Network) is a deep learning framework designed for multi-omics data integration and kidney cancer subtype classification.

![Model Overview](https://github.com/yoyolooki/DEGCN/blob/main/data/Figs1.png)

As shown in the figure, DEGCN takes multi-omics expression matrices as input, including but not limited to genomics, transcriptomics, and proteomics. The model leverages Graph Convolutional Networks (GCNs) and integrates two unsupervised multi-omics fusion algorithms:

- **Variational Autoencoder (VAE)**: Reduces the dimensionality of multi-omics expression data.
- **Similarity Network Fusion (SNF)**: Constructs a patient similarity network based on multi-omics profiles.

Feature extraction is not required before VAE and SNF, making DEGCN an efficient and streamlined pipeline for multi-omics data analysis.

---

## Installation

### Step 1: Set Up Environment
First, create a new Conda environment and install the required dependencies:

```bash
# Create a new Conda environment
conda create --name DEGCN python=3.8
conda activate DEGCN

# Clone the repository
git clone https://github.com/yoyolooki/DEGCN.git
cd DEGCN

# Install dependencies
pip install -r requirements.txt
```

---

## Data Preparation

### Step 2: Preprocess Data

You can either preprocess your own data using the provided R scripts or download the preprocessed datasets directly.

- Run the R script as described in `./data_processing/readme.md` to process raw data.
- Alternatively, download preprocessed data from https://figshare.com/articles/thesis/DEGCN-data/28517558.

---

## Running the Model

### Step 3: Execute the Pipeline

The DEGCN workflow consists of three main steps:

#### 1. Dimensionality Reduction using VAE
Reduce the dimensionality of multi-omics data using a Variational Autoencoder (VAE):

```bash
python VAE_run.py -p1 data/KCdata/fpkm.csv -p2 data/KCdata/gistic.csv -p3 data/KCdata/rppa.csv -s 0 -d gpu -e 100 -m 0 -bs 16
```

#### 2. Construct Patient Similarity Network using SNF
Build a patient similarity network using SNF:

```bash
python SNF.py -p KCdata/fpkm.csv KCdata/gistic.csv KCdata/rppa.csv -m sqeuclidean
```

#### 3. Train the Dense GCN Model
Use the learned multi-omics feature matrix and patient similarity network to train the Dense GCN model:

```bash
python DenseGCN_run.py -fd result/latent_data.csv -ad result/SNF_fused_matrix.csv -ld data/KCdata/sample_classes.csv -ts KCdata/test_sample.csv -m 0 -d gpu -p 20
```

---

## Contact
For any questions or support, please contact:

**Yu Li**  
📧 Email: [2023215215005@stu.ahtcm.edu.cn](mailto:2023215215005@stu.ahtcm.edu.cn)

---

## Citation
If you find DEGCN useful for your research, please consider citing our work:

*(Citation details to be added)*

---

## License
📜 **MIT License** © 2025 [Yu Li]