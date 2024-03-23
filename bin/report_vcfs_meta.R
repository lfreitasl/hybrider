#!/usr/bin/env Rscript

##Script for producing metadata to generated vcfs from dartR
library(dartR)
library(vcfR)

filenames<-list.files("./",pattern = "\\.vcf$")
files<-list.files("./",pattern = "\\.vcf$",full.names = T)
vcfs<-lapply(files, read.vcfR)
gls<-lapply(vcfs, vcfR2genlight)
n_locs<-sapply(gls, function(x) x$n.loc)
n_inds<-sapply(gls, function(x) length(x$ind.names))

#Making dataframe to export
vcf_meta<-data.frame(filenames,n_inds,n_locs)

#Exporting information
write.csv(vcf_meta,"./vcfs_info.csv",row.names = F,quote = F)

