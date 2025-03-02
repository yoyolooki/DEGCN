library(data.table)
library(dplyr)
library(tidyverse)

# Read KIRC's three-omics data and probe data
KIRC_rppa<-fread('data/KIRC/KIRC-RPPA', sep = '\t', header = TRUE, data.table = F)
KIRC_gistic<-fread('data/KIRC/TCGA-KIRC.gistic.tsv', sep = '\t', header = TRUE, data.table = F)
KIRC_fpkm<-fread('data/KIRC/TCGA-KIRC.htseq_fpkm.tsv', sep = '\t', header = TRUE, data.table = F)
KIRC_sample_genID<-fread('data/KICH/gencode.v22.annotation.gene.probeMap', sep = '\t', header = TRUE, data.table = F)
KIRC_sample_genID<-KIRC_sample_genID[,c(1,2)]

# Gistic and fpkm probe comparison
columns_to_keep <- !duplicated(colnames(KIRC_gistic))
KIRC_gistic <- KIRC_gistic[, columns_to_keep]
KIRC_gistic<-merge(KIRC_gistic, KIRC_sample_genID, by.x ="Gene Symbol", by.y = "id" )
sum(duplicated(KIRC_gistic$gene))
KIRC_gistic<-distinct(KIRC_gistic, gene, .keep_all = T)
sum(duplicated(KIRC_gistic$gene)) 
KIRC_gistic<-column_to_rownames(KIRC_gistic, "gene")
KIRC_gistic<-KIRC_gistic[,-1]

KIRC_fpkm<-merge(KIRC_fpkm, KIRC_sample_genID, by.x ="Ensembl_ID", by.y = "id" )
sum(duplicated(KIRC_fpkm$gene))
KIRC_fpkm<-distinct(KIRC_fpkm, gene, .keep_all = T)
sum(duplicated(KIRC_fpkm$gene)) 
KIRC_fpkm<-column_to_rownames(KIRC_fpkm, "gene")
KIRC_fpkm<-KIRC_fpkm[,-1]

KIRC_rppa<-column_to_rownames(KIRC_rppa, "sample")


# Get each group's column name
KIRC_rppa_colnames<-colnames(KIRC_rppa)

KIRC_gistic_colnames<-colnames(KIRC_gistic)
KIRC_gistic_colnames <- substr(KIRC_gistic_colnames, 1, nchar(KIRC_gistic_colnames) - 1)
colnames(KIRC_gistic)<-KIRC_gistic_colnames

KIRC_fpkm_colnames<-colnames(KIRC_fpkm)
KIRC_fpkm_colnames <- substr(KIRC_fpkm_colnames, 1, nchar(KIRC_fpkm_colnames) - 1)
colnames(KIRC_fpkm)<-KIRC_fpkm_colnames

# Take the intersection of the column names to get the final KIRC available sample name
KIRC_colnames_intersection_1<-intersect(KIRC_rppa_colnames, KIRC_gistic_colnames)
KIRC_colnames_intersection_2<-intersect(KIRC_colnames_intersection_1, KIRC_fpkm_colnames)
length(KIRC_colnames_intersection_2)  #469个
KIRC_samples<-KIRC_colnames_intersection_2


filter_KIRC_rppa<-KIRC_rppa[,KIRC_samples]
filter_KIRC_gistic<-KIRC_gistic[,KIRC_samples]
filter_KIRC_fpkm<-KIRC_fpkm[,KIRC_samples]

save(filter_KIRC_rppa, filter_KIRC_gistic, filter_KIRC_fpkm,file = "KIRC.Rdata")
