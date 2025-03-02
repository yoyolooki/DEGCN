library(data.table)
library(dplyr)
library(tidyverse)

# #Read KICH's three omics data and probe data
KICH_rppa<-fread('data/KICH/KICH-RPPA', sep = '\t', header = TRUE, data.table = F)
KICH_gistic<-fread('data/KICH/TCGA-KICH.gistic.tsv', sep = '\t', header = TRUE, data.table = F)
KICH_fpkm<-fread('data/KICH/TCGA-KICH.htseq_fpkm.tsv', sep = '\t', header = TRUE, data.table = F)
KICH_sample_genID<-fread('data/KICH/gencode.v22.annotation.gene.probeMap', sep = '\t', header = TRUE, data.table = F)
KICH_sample_genID<-KICH_sample_genID[,c(1,2)]

# Gistic and fpkm probe comparison
KICH_gistic<-merge(KICH_gistic, KICH_sample_genID, by.x ="Gene Symbol", by.y = "id" )
sum(duplicated(KICH_gistic$gene))  # Duplicate genes：84
KICH_gistic<-distinct(KICH_gistic, gene, .keep_all = T)  # Deletion of duplicate genes
sum(duplicated(KICH_gistic$gene)) 
KICH_gistic<-column_to_rownames(KICH_gistic, "gene")  # Set the gene column as row name
KICH_gistic<-KICH_gistic[,-1]  # Delete the extra first column

KICH_fpkm<-merge(KICH_fpkm, KICH_sample_genID, by.x ="Ensembl_ID", by.y = "id" )
sum(duplicated(KICH_fpkm$gene))  # Duplicate genes：2096
KICH_fpkm<-distinct(KICH_fpkm, gene, .keep_all = T)  # Deletion of duplicate genes
sum(duplicated(KICH_fpkm$gene)) 
KICH_fpkm<-column_to_rownames(KICH_fpkm, "gene")  # Set the gene column as row name
KICH_fpkm<-KICH_fpkm[,-1]  # Delete the extra first column

KICH_rppa<-column_to_rownames(KICH_rppa, "sample")  # The protein name column of rppa is set as the row name


# Take each group's column name
KICH_rppa_colnames<-colnames(KICH_rppa)

KICH_gistic_colnames<-colnames(KICH_gistic)
KICH_gistic_colnames <- substr(KICH_gistic_colnames, 1, nchar(KICH_gistic_colnames) - 1)
colnames(KICH_gistic)<-KICH_gistic_colnames

KICH_fpkm_colnames<-colnames(KICH_fpkm)
KICH_fpkm_colnames <- substr(KICH_fpkm_colnames, 1, nchar(KICH_fpkm_colnames) - 1)
colnames(KICH_fpkm)<-KICH_fpkm_colnames

# Take the intersection of the column names to get the final available sample name of KICH
KICH_colnames_intersection_1<-intersect(KICH_rppa_colnames, KICH_gistic_colnames)
KICH_colnames_intersection_2<-intersect(KICH_colnames_intersection_1, KICH_fpkm_colnames)
length(KICH_colnames_intersection_2)  #63个
KICH_samples<-KICH_colnames_intersection_2


# Obtain the three omics data of KICH after processing the sample control
filter_KICH_rppa<-KICH_rppa[,KICH_samples]
filter_KICH_gistic<-KICH_gistic[,KICH_samples]
filter_KICH_fpkm<-KICH_fpkm[,KICH_samples]


save(filter_KICH_rppa, filter_KICH_gistic, filter_KICH_fpkm,file = "KICH.Rdata")
