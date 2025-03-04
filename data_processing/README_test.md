# Renal Cancer Multi-Omics Data Processing

This repository provides R scripts for processing and integrating multi-omics data (RPPA, GISTIC, FPKM) across three renal cancer subtypes:  
**KICH** (Kidney Chromophobe), **KIRC** (Kidney Renal Clear Cell Carcinoma), and **KIRP** (Kidney Renal Papillary Cell Carcinoma).



## Script Overview

### **1. Subtype-Specific Processing**
| Script       | Functionality                                                                                     | Output File                                           |
|--------------|---------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| `KICH.R`     | Processes omics data for Kidney Chromophobe (KICH)                                                | `KICH.Rdata`                                          |
| `KIRC.R`     | Processes omics data for Kidney Renal Clear Cell Carcinoma (KIRC)                                 | `KIRC.Rdata`                                          |
| `KIRP.R`     | Processes omics data for Kidney Renal Papillary Cell Carcinoma (KIRP)                             | `KIRP.Rdata`                                          |
| `combined.R` | Multiple omics data (FPKM, Gistic, RPPA) of three renal cancer subtypes were merged and processed | `fpkm.csv` `gistic.csv` `rppa.csv` `sample_class.csv` |

## Installation

### **Dependencies**
Install required R packages:
```R
install.packages(c("data.table", "dplyr", "tidyverse"))
```

Data Preparation
Download data from Figshare: https://figshare.com/articles/thesis/DEGCN-data/28517558

🚀 Usage Run a Single Subtype Script

```R
# Process KICH data
source("scripts/KICH.R")  # Output: KICH.Rdata
# Load processed data
load("output/KICH.Rdata")  # Assign to variable (e.g., kich_data)
```

Run Integration Script

```R
source("combined.R")  # Output: fpkm.csv gistic.csv rppa.csv sample_class.csv
```


📝 Notes

Raw data files must follow the naming convention: RPPA.csv, GISTIC.csv, FPKM.csv within each subtype folder.

For reproducibility, use R version ≥4.0.

📜 License
MIT License © 2025 [YU LI]


---

