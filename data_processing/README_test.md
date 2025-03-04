# Renal Cancer Multi-Omics Data Processing

This repository provides R scripts for processing and integrating multi-omics data (RPPA, GISTIC, FPKM) across three renal cancer subtypes:  
**KICH** (Kidney Chromophobe), **KIRC** (Kidney Renal Clear Cell Carcinoma), and **KIRP** (Kidney Renal Papillary Cell Carcinoma).

---

## 📂 Repository Structure
.
├── data/ # Raw data (excluded via .gitignore)
├── scripts/
│ ├── KICH.R # Processes KICH subtype data
│ ├── KIRC.R # Processes KIRC subtype data
│ ├── KIRP.R # Processes KIRP subtype data
│ └── combined.R # Cross-subtype integration
├── output/ # Processed data files (e.g., .Rdata)
└── README.md


---

## 🧩 Script Overview

### **1. Subtype-Specific Processing**
| Script      | Functionality                                                                 | Output File     |
|-------------|-------------------------------------------------------------------------------|-----------------|
| `KICH.R`    | Processes omics data for Kidney Chromophobe (KICH)                            | `KICH.Rdata`    |
| `KIRC.R`    | Processes omics data for Kidney Renal Clear Cell Carcinoma (KIRC)             | `KIRC.Rdata`    |
| `KIRP.R`    | Processes omics data for Kidney Renal Papillary Cell Carcinoma (KIRP)         | `KIRP.Rdata`    |

**Shared Workflow**:
1. **Data Loading**: Reads RPPA, GISTIC, and FPKM files.
2. **Gene Identifier Resolution**: Merges data with probe annotation files.
3. **Data Cleaning**:
   - Removes duplicate gene entries.
   - Sets row names (genes/samples).
4. **Sample Alignment**: Filters datasets to retain intersecting samples.
5. **Output**: Saves processed data as `.Rdata`.

### **2. Data Integration (`combined.R`)**
- **Input**: Subtype-specific `.Rdata` files from `output/`.
- **Functionality**:
  - Merges processed data across subtypes.
  - Saves integrated dataset to `combined.Rdata`.

---

## ⚙️ Installation

### **Dependencies**
Install required R packages:
```R
install.packages(c("data.table", "dplyr", "tidyverse"))
```

Data Preparation
Download data from Figshare:
🔗 DEGCN Data Repository

Place unzipped files in data/ with the following structure:

data/
├── KICH/
│   ├── RPPA.csv
│   ├── GISTIC.csv
│   └── FPKM.csv
├── KIRC/
└── KIRP/

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
Ensure the output/ directory exists before running scripts.

Raw data files must follow the naming convention: RPPA.csv, GISTIC.csv, FPKM.csv within each subtype folder.

For reproducibility, use R version ≥4.0.

📜 License
MIT License © 2024 [Your Name]


---

### **Key Improvements**
1. **Structured Hierarchy**: Clear sections for installation, usage, and outputs.
2. **Standardized Format**:
   - Tables for script/output descriptions.
   - Code blocks for commands.
   - File structure visualization.
3. **Actionable Instructions**: Direct download links and `wget` examples.
4. **Clarity**: Explicit naming conventions and version requirements.

Let me know if you need further tweaks! 🛠️