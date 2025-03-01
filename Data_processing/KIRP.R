library(data.table)
library(dplyr)
library(tidyverse)

#读KIRP的三个组学数据和探针数据
KIRP_rppa<-fread('data/KIRP/KIRP-RPPA', sep = '\t', header = TRUE, data.table = F)
KIRP_gistic<-fread('data/KIRP/TCGA-KIRP.gistic.tsv', sep = '\t', header = TRUE, data.table = F)
KIRP_fpkm<-fread('data/KIRP/TCGA-KIRP.htseq_fpkm.tsv', sep = '\t', header = TRUE, data.table = F)
KIRP_sample_genID<-fread('data/KICH/gencode.v22.annotation.gene.probeMap', sep = '\t', header = TRUE, data.table = F)
KIRP_sample_genID<-KIRP_sample_genID[,c(1,2)]

#gistic、fpkm探针对照
columns_to_keep <- !duplicated(colnames(KIRP_gistic)) # 创建一个逻辑向量，指示哪些列应该被保留（保留第一次出现的列）
KIRP_gistic <- KIRP_gistic[, columns_to_keep]  #删除KIRC_gistic的重复列

KIRP_gistic<-merge(KIRP_gistic, KIRP_sample_genID, by.x ="Gene Symbol", by.y = "id" )
sum(duplicated(KIRP_gistic$gene))  #重复基因：84
KIRP_gistic<-distinct(KIRP_gistic, gene, .keep_all = T)  #删除重复基因
sum(duplicated(KIRP_gistic$gene)) 
KIRP_gistic<-column_to_rownames(KIRP_gistic, "gene")  #gene列设为行名
KIRP_gistic<-KIRP_gistic[,-1]  #删除多余的第一列

KIRP_fpkm<-merge(KIRP_fpkm, KIRP_sample_genID, by.x ="Ensembl_ID", by.y = "id" )
sum(duplicated(KIRP_fpkm$gene))  #重复基因：2096
KIRP_fpkm<-distinct(KIRP_fpkm, gene, .keep_all = T)  #删除重复基因
sum(duplicated(KIRP_fpkm$gene)) 
KIRP_fpkm<-column_to_rownames(KIRP_fpkm, "gene")  #gene列设为行名
KIRP_fpkm<-KIRP_fpkm[,-1]  #删除多余的第一列

KIRP_rppa<-column_to_rownames(KIRP_rppa, "sample")  #rppa的蛋白质名列设为行名


#拿每个组学列名
KIRP_rppa_colnames<-colnames(KIRP_rppa)

KIRP_gistic_colnames<-colnames(KIRP_gistic)
KIRP_gistic_colnames <- substr(KIRP_gistic_colnames, 1, nchar(KIRP_gistic_colnames) - 1)  #删掉样本名最后多的一个字母
colnames(KIRP_gistic)<-KIRP_gistic_colnames #将修改过的列名赋给数据框

KIRP_fpkm_colnames<-colnames(KIRP_fpkm)
KIRP_fpkm_colnames <- substr(KIRP_fpkm_colnames, 1, nchar(KIRP_fpkm_colnames) - 1)  #删掉样本名最后多的一个字母
colnames(KIRP_fpkm)<-KIRP_fpkm_colnames  #将修改过的列名赋给数据框

#列名取交集，得到最终KIRP的可用样本名
KIRP_colnames_intersection_1<-intersect(KIRP_rppa_colnames, KIRP_gistic_colnames)
KIRP_colnames_intersection_2<-intersect(KIRP_colnames_intersection_1, KIRP_fpkm_colnames)
length(KIRP_colnames_intersection_2)  #213个
KIRP_samples<-KIRP_colnames_intersection_2


#得到处理好样本对照的KIRP的三个组学数据
filter_KIRP_rppa<-KIRP_rppa[,KIRP_samples]
filter_KIRP_gistic<-KIRP_gistic[,KIRP_samples]
filter_KIRP_fpkm<-KIRP_fpkm[,KIRP_samples]


#保存
save(filter_KIRP_rppa, filter_KIRP_gistic, filter_KIRP_fpkm,file = "KIRP.Rdata")
