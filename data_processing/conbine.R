library(dplyr)

load('KICH.Rdata')
load('KIRC.Rdata')
load('KIRP.Rdata')


# Merge fpkm
filter_KICH_fpkm<-t(filter_KICH_fpkm)  # Transpose
filter_KIRC_fpkm<-t(filter_KIRC_fpkm)
filter_KIRP_fpkm<-t(filter_KIRP_fpkm)
fpkm=rbind(filter_KICH_fpkm, filter_KIRC_fpkm, filter_KIRP_fpkm)
fpkm <- as.data.frame(fpkm)
fpkm$sample<-rownames(fpkm)
rownames(fpkm) <- NULL
fpkm <- fpkm[, c(ncol(fpkm), 1:(ncol(fpkm) - 1))]
any(is.na(fpkm))  # Check if there are missing values: FALSE

# Merge gistic
filter_KICH_gistic<-t(filter_KICH_gistic)
filter_KIRC_gistic<-t(filter_KIRC_gistic)
filter_KIRP_gistic<-t(filter_KIRP_gistic)
gistic=rbind(filter_KICH_gistic, filter_KIRC_gistic, filter_KIRP_gistic)
gistic <- as.data.frame(gistic)
gistic$sample<-rownames(gistic)
rownames(gistic) <- NULL
gistic <- gistic[, c(ncol(gistic), 1:(ncol(gistic) - 1))]
any(is.na(gistic))  #FALSE


# Merge rppa
filter_KICH_rppa<-t(filter_KICH_rppa)
filter_KIRC_rppa<-t(filter_KIRC_rppa)
filter_KIRP_rppa<-t(filter_KIRP_rppa)


# #rppa protein name data is different and needs additional processing

KICH_rppa_colnames<-colnames(filter_KICH_rppa)
KIRC_rppa_colnames<-colnames(filter_KIRC_rppa)
KIRP_rppa_colnames<-colnames(filter_KIRP_rppa)

intersection_1<-intersect(KICH_rppa_colnames, KIRC_rppa_colnames)
intersection_2<-intersect(intersection_1, KIRP_rppa_colnames)
length(intersection_2)  #63
all_pro<-intersection_2

filter_KICH_rppa<-filter_KICH_rppa[,all_pro]
filter_KIRC_rppa<-filter_KIRC_rppa[,all_pro]
filter_KIRP_rppa<-filter_KIRP_rppa[,all_pro]
rppa=rbind(filter_KICH_rppa, filter_KIRC_rppa, filter_KIRP_rppa)
rppa <- as.data.frame(rppa)
rppa$sample<-rownames(rppa)
rownames(rppa) <- NULL
rppa <- rppa[, c(ncol(rppa), 1:(ncol(rppa) - 1))]


any(is.na(rppa))  #TRUE
rppa <- rppa %>% mutate(across(where(is.numeric), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))
any(is.na(rppa)) #FALSE

# Save Rdata
save(fpkm, gistic, rppa, file = 'conbined_data.Rdata')


# Save training data
write.csv(fpkm, file = 'fpkm.csv', quote = F, row.names = F)
write.csv(gistic, file = 'gistic.csv', quote = F, row.names = F)
write.csv(rppa, file = 'rppa.csv', quote = F, row.names = F)

# #Generate and save label data
dim(filter_KICH_fpkm)  #63 58387
dim(filter_KIRC_fpkm) #469 58387
dim(filter_KIRP_fpkm) #213 58387

sample<-c(rownames(fpkm))
label<-c(rep('0',63),rep('1',469),rep('2',213))
cancer_subtype<-c(rep('KICH',63),rep('KIRC',469),rep('KIRP',213))
sample_datafram<-data.frame(sample,label,cancer_subtype)
write.csv(sample_datafram, file = 'sample_class.csv',row.names = F, , quote = F)
