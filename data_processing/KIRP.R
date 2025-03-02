library(data.table)
library(dplyr)
library(tidyverse)

# Read KIRP's three-omics data and probe data
KIRP_rppa<-fread('data/KIRP/KIRP-RPPA', sep = '\t', header = TRUE, data.table = F)
KIRP_gistic<-fread('data/KIRP/TCGA-KIRP.gistic.tsv', sep = '\t', header = TRUE, data.table = F)
KIRP_fpkm<-fread('data/KIRP/TCGA-KIRP.htseq_fpkm.tsv', sep = '\t', header = TRUE, data.table = F)
KIRP_sample_genID<-fread('data/KICH/gencode.v22.annotation.gene.probeMap', sep = '\t', header = TRUE, data.table = F)
KIRP_sample_genID<-KIRP_sample_genID[,c(1,2)]

# Gistic and fpkm probe comparison
columns_to_keep <- !duplicated(colnames(KIRP_gistic))
KIRP_gistic <- KIRP_gistic[, columns_to_keep]

KIRP_gistic<-merge(KIRP_gistic, KIRP_sample_genID, by.x ="Gene Symbol", by.y = "id" )
sum(duplicated(KIRP_gistic$gene))
KIRP_gistic<-distinct(KIRP_gistic, gene, .keep_all = T)
sum(duplicated(KIRP_gistic$gene)) 
KIRP_gistic<-column_to_rownames(KIRP_gistic, "gene")
KIRP_gistic<-KIRP_gistic[,-1]

KIRP_fpkm<-merge(KIRP_fpkm, KIRP_sample_genID, by.x ="Ensembl_ID", by.y = "id" )
sum(duplicated(KIRP_fpkm$gene))
KIRP_fpkm<-distinct(KIRP_fpkm, gene, .keep_all = T)
sum(duplicated(KIRP_fpkm$gene)) 
KIRP_fpkm<-column_to_rownames(KIRP_fpkm, "gene")
KIRP_fpkm<-KIRP_fpkm[,-1]

KIRP_rppa<-column_to_rownames(KIRP_rppa, "sample")


# Get each group's column name
KIRP_rppa_colnames<-colnames(KIRP_rppa)

KIRP_gistic_colnames<-colnames(KIRP_gistic)
KIRP_gistic_colnames <- substr(KIRP_gistic_colnames, 1, nchar(KIRP_gistic_colnames) - 1)
colnames(KIRP_gistic)<-KIRP_gistic_colnames

KIRP_fpkm_colnames<-colnames(KIRP_fpkm)
KIRP_fpkm_colnames <- substr(KIRP_fpkm_colnames, 1, nchar(KIRP_fpkm_colnames) - 1)
colnames(KIRP_fpkm)<-KIRP_fpkm_colnames

# Take the intersection of the column names to get the final KIRP available sample name
KIRP_colnames_intersection_1<-intersect(KIRP_rppa_colnames, KIRP_gistic_colnames)
KIRP_colnames_intersection_2<-intersect(KIRP_colnames_intersection_1, KIRP_fpkm_colnames)
length(KIRP_colnames_intersection_2)  #213个
KIRP_samples<-KIRP_colnames_intersection_2

# Get the three omics data of KIRP after processing the sample control
filter_KIRP_rppa<-KIRP_rppa[,KIRP_samples]
filter_KIRP_gistic<-KIRP_gistic[,KIRP_samples]
filter_KIRP_fpkm<-KIRP_fpkm[,KIRP_samples]

save(filter_KIRP_rppa, filter_KIRP_gistic, filter_KIRP_fpkm,file = "KIRP.Rdata")
