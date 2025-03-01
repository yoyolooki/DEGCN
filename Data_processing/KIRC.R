library(data.table)
library(dplyr)
library(tidyverse)

#读KIRC的三个组学数据和探针数据
KIRC_rppa<-fread('data/KIRC/KIRC-RPPA', sep = '\t', header = TRUE, data.table = F)
KIRC_gistic<-fread('data/KIRC/TCGA-KIRC.gistic.tsv', sep = '\t', header = TRUE, data.table = F)
KIRC_fpkm<-fread('data/KIRC/TCGA-KIRC.htseq_fpkm.tsv', sep = '\t', header = TRUE, data.table = F)
KIRC_sample_genID<-fread('data/KICH/gencode.v22.annotation.gene.probeMap', sep = '\t', header = TRUE, data.table = F)
KIRC_sample_genID<-KIRC_sample_genID[,c(1,2)]

#gistic、fpkm探针对照
columns_to_keep <- !duplicated(colnames(KIRC_gistic)) # 创建一个逻辑向量，指示哪些列应该被保留（保留第一次出现的列）
KIRC_gistic <- KIRC_gistic[, columns_to_keep]  #删除KIRC_gistic的重复列
KIRC_gistic<-merge(KIRC_gistic, KIRC_sample_genID, by.x ="Gene Symbol", by.y = "id" )
sum(duplicated(KIRC_gistic$gene))  #重复基因：84
KIRC_gistic<-distinct(KIRC_gistic, gene, .keep_all = T)  #删除重复基因
sum(duplicated(KIRC_gistic$gene)) 
KIRC_gistic<-column_to_rownames(KIRC_gistic, "gene")  #gene列设为行名
KIRC_gistic<-KIRC_gistic[,-1]  #删除多余的第一列

KIRC_fpkm<-merge(KIRC_fpkm, KIRC_sample_genID, by.x ="Ensembl_ID", by.y = "id" )
sum(duplicated(KIRC_fpkm$gene))  #重复基因：2096
KIRC_fpkm<-distinct(KIRC_fpkm, gene, .keep_all = T)  #删除重复基因
sum(duplicated(KIRC_fpkm$gene)) 
KIRC_fpkm<-column_to_rownames(KIRC_fpkm, "gene")  #gene列设为行名
KIRC_fpkm<-KIRC_fpkm[,-1]  #删除多余的第一列

KIRC_rppa<-column_to_rownames(KIRC_rppa, "sample")  #rppa的蛋白质名列设为行名


#拿每个组学列名
KIRC_rppa_colnames<-colnames(KIRC_rppa)

KIRC_gistic_colnames<-colnames(KIRC_gistic)
KIRC_gistic_colnames <- substr(KIRC_gistic_colnames, 1, nchar(KIRC_gistic_colnames) - 1)  #删掉样本名最后多的一个字母
colnames(KIRC_gistic)<-KIRC_gistic_colnames #将修改过的列名赋给数据框

KIRC_fpkm_colnames<-colnames(KIRC_fpkm)
KIRC_fpkm_colnames <- substr(KIRC_fpkm_colnames, 1, nchar(KIRC_fpkm_colnames) - 1)  #删掉样本名最后多的一个字母
colnames(KIRC_fpkm)<-KIRC_fpkm_colnames  #将修改过的列名赋给数据框

#列名取交集，得到最终KIRC的可用样本名
KIRC_colnames_intersection_1<-intersect(KIRC_rppa_colnames, KIRC_gistic_colnames)
KIRC_colnames_intersection_2<-intersect(KIRC_colnames_intersection_1, KIRC_fpkm_colnames)
length(KIRC_colnames_intersection_2)  #469个
KIRC_samples<-KIRC_colnames_intersection_2


filter_KIRC_rppa<-KIRC_rppa[,KIRC_samples]
filter_KIRC_gistic<-KIRC_gistic[,KIRC_samples]
filter_KIRC_fpkm<-KIRC_fpkm[,KIRC_samples]

save(filter_KIRC_rppa, filter_KIRC_gistic, filter_KIRC_fpkm,file = "KIRC.Rdata")
