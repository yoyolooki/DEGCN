# Renal Cancer Multi-Omics Data Processing

This repository provides R scripts for processing and integrating multi-omics data (RPPA, GISTIC, FPKM) across three renal cancer subtypes:  
**KICH** (Kidney Chromophobe), **KIRC** (Kidney Renal Clear Cell Carcinoma), and **KIRP** (Kidney Renal Papillary Cell Carcinoma).



## Script Overview

### **1. Subtype-Specific Processing**
| Script       | Functionality                                                                                     | Output File                         |
|--------------|---------------------------------------------------------------------------------------------------|-------------------------------------|
| `KICH.R`     | Processes omics data for Kidney Chromophobe (KICH)                                                | `KICH.Rdata`                        |
| `KIRC.R`     | Processes omics data for Kidney Renal Clear Cell Carcinoma (KIRC)                                 | `KIRC.Rdata`                        |
| `KIRP.R`     | Processes omics data for Kidney Renal Papillary Cell Carcinoma (KIRP)                             | `KIRP.Rdata`                        |
| `combined.R` | Multiple omics data (FPKM, Gistic, RPPA) of three renal cancer subtypes were merged and processed | `fpkm.csv` ,`gistic.csv`,`rppa.csv` |

**Shared Workflow**:
1. **Data Loading**: Reads RPPA, GISTIC, and FPKM files.
2. **Gene Identifier Resolution**: Merges data with probe annotation files.
3. **Data Cleaning**:
   - Removes duplicate gene entries.
   - Sets row names (genes/samples).
4. **Sample Alignment**: Filters datasets to retain intersecting samples.
5. **Output**: Saves processed data as `.Rdata`.

### **2. Data Integration (`combined.R`)**
- **Input**: Processed `.Rdata` files from subtype scripts (`KICH.Rdata`, `KIRC.Rdata`, `KIRP.Rdata`)
- **Key Operations**:
  1. **Data Merging**:
     - Transposes omics matrices to sample × feature format
     - Identifies common RPPA proteins across subtypes
  2. **Data Cleaning**:
     - Imputes missing RPPA values with feature means
     - Removes duplicate samples (FPKM only)
  3. **Label Generation**:
     - Creates unified sample labels (0=KICH, 1=KIRC, 2=KIRP)
- **Outputs**:
  - `combined.Rdata`: Integrated dataset (FPKM/GISTIC/RPPA)
  - CSV files: `fpkm.csv`, `gistic.csv`, `rppa.csv`, `sample_class.csv`

## Installation

### **Dependencies**
Install required R packages:
```R
install.packages(c("data.table", "dplyr", "tidyverse"))
```

Data Preparation
Download data from Figshare:
🔗 DEGCN Data Repository

🚀 Usage
Run a Single Subtype Script

```R
# Process KICH data
source("scripts/KICH.R")  # Output: output/KICH.Rdata

# Load processed data
load("output/KICH.Rdata")  # Assign to variable (e.g., kich_data)
```

Run Integration Script

```R
source("scripts/combined.R")  # Output: output/combined.Rdata
```

📊 Output Files


File	Description	Structure
KICH.Rdata	Processed omics data for KICH	List (RPPA, GISTIC, FPKM)
KIRC.Rdata	Processed omics data for KIRC	List (RPPA, GISTIC, FPKM)
KIRP.Rdata	Processed omics data for KIRP	List (RPPA, GISTIC, FPKM)
combined.Rdata	Integrated multi-subtype dataset	Matrix (Samples × Features)


📝 Notes

Raw data files must follow the naming convention: RPPA.csv, GISTIC.csv, FPKM.csv within each subtype folder.

For reproducibility, use R version ≥4.0.

📜 License
MIT License © 2025 [YU LI]


---

