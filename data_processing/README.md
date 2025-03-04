# Renal Cancer Multi-Omics Data Processing

## Overview

This repository provides R scripts for processing and integrating multi-omics data (RPPA, GISTIC, FPKM) across three
renal cancer subtypes:

- **KICH** (*Kidney Chromophobe*)
- **KIRC** (*Kidney Renal Clear Cell Carcinoma*)
- **KIRP** (*Kidney Renal Papillary Cell Carcinoma*)

These scripts facilitate the preprocessing, merging, and integration of multi-omics data to support downstream analysis,
such as classification tasks using DEGCN.

---

## Script Overview

### **1. Subtype-Specific Processing**

| Script       | Functionality                                                                            | Output Files                                                                         |
|--------------|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| `KICH.R`     | Processes omics data for **Kidney Chromophobe (KICH)**                                   | `output/KICH.Rdata`                                                                  |
| `KIRC.R`     | Processes omics data for **Kidney Renal Clear Cell Carcinoma (KIRC)**                    | `output/KIRC.Rdata`                                                                  |
| `KIRP.R`     | Processes omics data for **Kidney Renal Papillary Cell Carcinoma (KIRP)**                | `output/KIRP.Rdata`                                                                  |
| `combined.R` | Merges and processes **FPKM, GISTIC, and RPPA** data for all three renal cancer subtypes | `output/fpkm.csv`, `output/gistic.csv`, `output/rppa.csv`, `output/sample_class.csv` |

---

## Installation

### **1. Dependencies**

Install the required R packages before running the scripts:

```r
install.packages(c("data.table", "dplyr", "tidyverse"))
```

### **2. Data Preparation**

Download the original dataset from Figshare, including KICH.zip, KIRC.zip, and
KIRP.zip: https://figshare.com/articles/thesis/DEGCN-data/28517558

Ensure the raw data files are named correctly within their respective subtype folders:

```
├── data/
│   ├── KICH/
│   │   ├── gencode.v22.annotation.gene.probeMap
│   │   ├── KICH-RPPA
│   │   ├── TCGA-KICH.gistic.tsv
│   │   ├── TCGA-KICH.htseq_fpkm.tsv
│   ├── KIRC/
│   ├── KIRP/
```

---

## Usage

### **1. Run a Single Subtype Script**

```r
# Process KICH data
source("scripts/KICH.R")  # Generates output/KICH.Rdata

# Load processed data
load("output/KICH.Rdata")  # Assign to variable (e.g., kich_data)
```

### **2. Run the Integration Script**

Merge and process multi-omics data for all subtypes:

```r
source("scripts/combined.R")  # Generates fpkm.csv, gistic.csv, rppa.csv, sample_class.csv
```

---

## Notes

- Ensure raw data files are named as `RPPA.csv`, `GISTIC.csv`, and `FPKM.csv` in each subtype folder.
- For reproducibility, use **R version ≥ 4.3.3**.

---

## License

📜 **MIT License** © 2025 [Yu Li]