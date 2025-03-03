# Overview
This repository contains R scripts for processing and integrating omics data for three subtypes of renal cancer

# Scripts
1.KICH.R
* Description: Processes omics data for Kidney Chromophobe (KICH) cancer subtype.
* Functionality: 
  * Reads RPPA, GISTIC, and FPKM data files.
  * Merges data with gene probe information to resolve gene identifiers.
  * Removes duplicate gene entries.
  * Sets appropriate row names (genes or samples).
  * Finds the intersection of sample names across all datasets.
  * Filters datasets to include only common samples.
  * Saves the processed data to KICH.Rdata. 

2.KIRC.R
* Description: Processes omics data for Kidney Renal Clear Cell Carcinoma (KIRC) cancer subtype.
* Functionality: 
  * Similar to KICH.R, this script performs analogous data reading, merging, cleaning, and intersection steps.
  * Saves the processed data to KIRC.Rdata.
  
3.KIPR.R
* Decription: Processes omics data for Kidney Renal Papillary Cell Carcinoma (KIPR) cancer subtype.
* Functionality:
  * Analogous to the other scripts, it reads, merges, cleans, and intersects data.
  * Saves the processed data to KIRP.Rdata 

4.combined.R
* Description: Processes and integrates omics data across multiple cancer subtypes.
* Functionality:
  * Reads, merges, cleans, and intersects data from various cancer subtype scripts.
  * Consolidates processed data into a single file for comprehensive analysis.
  * Saves the integrated data to combined.Rdata
  
# Dependencies
* R Packages:
  * data.table
  * dplyr
  * tidyverse

# Notes
1.The input file of the R script can be downloaded from https://figshare.com/articles/thesis/DEGCN-data/28517558. (KICH.zip, KIRC.zip, KIRP.zip)
2.You can also download the data processed by the R script directly from above