# Omics Data Processing for Renal Cancer Subtypes

![Renal Cancer Subtypes](https://via.placeholder.com/800x200.png?text=Omics+Data+Integration+Workflow)  
*(建议添加项目示意图或流程图，替换此占位图)*

This repository provides R scripts for processing and integrating multi-omics data across three renal cancer subtypes:  
**KICH** (Kidney Chromophobe), **KIRC** (Kidney Renal Clear Cell Carcinoma), and **KIRP** (Kidney Renal Papillary Cell Carcinoma).

---

## 📁 Repository Structure

project-root/
├── data/ # Raw data (excluded via .gitignore)
├── scripts/
│ ├── KICH.R # KICH subtype processing
│ ├── KIRC.R # KIRC subtype processing
│ ├── KIRP.R # KIRP subtype processing
│ └── combined.R # Cross-subtype integration
├── output/ # Processed data files
│ ├── KICH.Rdata
│ ├── KIRC.Rdata
│ └── ...
└── README.md


---

## 🛠️ Scripts

### **1. Subtype-Specific Processing**
| Script      | Description                                                                 | Output File     |
|-------------|-----------------------------------------------------------------------------|-----------------|
| `KICH.R`    | Processes Kidney Chromophobe (KICH) omics data                              | `KICH.Rdata`    |
| `KIRC.R`    | Processes Kidney Renal Clear Cell Carcinoma (KIRC) data                     | `KIRC.Rdata`    |
| `KIRP.R`    | Processes Kidney Renal Papillary Cell Carcinoma (KIRP) data                 | `KIRP.Rdata`    |

**统一功能流程**:
1. **数据读取**: RPPA, GISTIC, FPKM 原始数据
2. **基因标识解析**: 合并基因探针信息
3. **数据清洗**:
   - 移除重复基因条目
   - 规范行名（基因/样本）
4. **样本对齐**: 跨数据集样本交集筛选
5. **保存**: 标准化输出为 `.Rdata` 文件

### **2. Combined Integration (`combined.R`)**
- **输入**: 各亚型处理后的 `*.Rdata` 文件
- **功能**:
  - 跨亚型数据整合
  - 批次效应校正 (建议添加具体方法)
  - 保存整合数据至 `combined.Rdata`

---

## ⚙️ 安装与依赖

### **R 包依赖**
```r
install.packages(c("data.table", "dplyr", "tidyverse"))