library(data.table)
library(dplyr)
library(tidyverse)

#读KICH的三个组学数据和探针数据
KICH_rppa<-fread('data/KICH/KICH-RPPA', sep = '\t', header = TRUE, data.table = F)
KICH_gistic<-fread('data/KICH/TCGA-KICH.gistic.tsv', sep = '\t', header = TRUE, data.table = F)
KICH_fpkm<-fread('data/KICH/TCGA-KICH.htseq_fpkm.tsv', sep = '\t', header = TRUE, data.table = F)
KICH_sample_genID<-fread('data/KICH/gencode.v22.annotation.gene.probeMap', sep = '\t', header = TRUE, data.table = F)
KICH_sample_genID<-KICH_sample_genID[,c(1,2)]

#gistic、fpkm探针对照
KICH_gistic<-merge(KICH_gistic, KICH_sample_genID, by.x ="Gene Symbol", by.y = "id" )
sum(duplicated(KICH_gistic$gene))  #重复基因：84
KICH_gistic<-distinct(KICH_gistic, gene, .keep_all = T)  #删除重复基因
sum(duplicated(KICH_gistic$gene)) 
KICH_gistic<-column_to_rownames(KICH_gistic, "gene")  #gene列设为行名
KICH_gistic<-KICH_gistic[,-1]  #删除多余的第一列

KICH_fpkm<-merge(KICH_fpkm, KICH_sample_genID, by.x ="Ensembl_ID", by.y = "id" )
sum(duplicated(KICH_fpkm$gene))  #重复基因：2096
KICH_fpkm<-distinct(KICH_fpkm, gene, .keep_all = T)  #删除重复基因
sum(duplicated(KICH_fpkm$gene)) 
KICH_fpkm<-column_to_rownames(KICH_fpkm, "gene")  #gene列设为行名
KICH_fpkm<-KICH_fpkm[,-1]  #删除多余的第一列

KICH_rppa<-column_to_rownames(KICH_rppa, "sample")  #rppa的蛋白质名列设为行名


#拿每个组学列名
KICH_rppa_colnames<-colnames(KICH_rppa)

KICH_gistic_colnames<-colnames(KICH_gistic)
KICH_gistic_colnames <- substr(KICH_gistic_colnames, 1, nchar(KICH_gistic_colnames) - 1)  #删掉样本名最后多的一个字母
colnames(KICH_gistic)<-KICH_gistic_colnames #将修改过的列名赋给数据框

KICH_fpkm_colnames<-colnames(KICH_fpkm)
KICH_fpkm_colnames <- substr(KICH_fpkm_colnames, 1, nchar(KICH_fpkm_colnames) - 1)  #删掉样本名最后多的一个字母
colnames(KICH_fpkm)<-KICH_fpkm_colnames  #将修改过的列名赋给数据框

#列名取交集，得到最终KICH的可用样本名
KICH_colnames_intersection_1<-intersect(KICH_rppa_colnames, KICH_gistic_colnames)
KICH_colnames_intersection_2<-intersect(KICH_colnames_intersection_1, KICH_fpkm_colnames)
length(KICH_colnames_intersection_2)  #63个
KICH_samples<-KICH_colnames_intersection_2


#得到处理好样本对照的KICH的三个组学数据
filter_KICH_rppa<-KICH_rppa[,KICH_samples]
filter_KICH_gistic<-KICH_gistic[,KICH_samples]
filter_KICH_fpkm<-KICH_fpkm[,KICH_samples]


#保存
save(filter_KICH_rppa, filter_KICH_gistic, filter_KICH_fpkm,file = "KICH.Rdata")
