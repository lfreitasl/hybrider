#!/usr/bin/env Rscript

##Script for gl object filtering and conversion to various formats: Treemix, snapp, structure, VCF, bayescan and svdquartets.
library(dartR, quietly=T)
library(vcfR, quietly=T)

args<-commandArgs(trailingOnly=TRUE)

vcfp<-args[1] #Caminho do vcf
meta<-args[2] #Planilha de metadados
loci_filt_thresh<-as.numeric(args[3]) #filt loci based on missing threshold
ind_filt_thresh<-as.numeric(args[4])  #filt inds based on missing threshold
maf<-as.numeric(args[5])  #filt loci based on minor allele frequency
pref<-args[6] #prefix for naming the outputs
usepopinfo<-as.logical(args[7]) #whether or not use population information if present on samplesheet

#Reading vcf into genlight object
myg<-gl.read.vcf(vcfp)

#reading metadata table and matching the order of column "samples" in metadata to order of inds in gl
meta<-read.table(meta, header=T)
meta<-meta[meta$samples%in%myg$ind.names,]
meta<-meta[match(myg$ind.names, meta$samples),]

# Adding filters
myg<-gl.filter.allna(myg)
myg<-gl.filter.monomorphs(myg)
myg<-gl.filter.callrate(myg, threshold = loci_filt_thresh)
myg<-gl.filter.callrate(myg, method = "ind", threshold = ind_filt_thresh)
myg<-gl.filter.maf(myg,threshold = maf)

#Adding pop information to a new gl object
if(usepopinfo){
myg_p<-myg
myg_p$pop<-as.factor(meta$POP)

# Adding filters to gl with pop info
myg_p<-gl.filter.allna(myg_p, by.pop = T)
myg_p<-gl.filter.monomorphs(myg_p)
myg_p<-gl.filter.callrate(myg_p, threshold = loci_filt_thresh)
myg_p<-gl.filter.callrate(myg_p, method = "ind", threshold = ind_filt_thresh)
myg_p<-gl.filter.maf(myg_p,threshold = maf,by.pop = T, ind.limit = 4)

#for treemix
gl2treemix(myg_p, outfile = paste(pref, ".treemix.gz", sep = ""), outpath = "./")

}
# Adding filter to remove some populations if needed:
#myg_p<-gl.drop.pop(myg_p, c("-","Cativeiro","Japi","Uruçui")) #substitute concatenated per samples
#myg_p<-gl.filter.allna(myg_p, by.pop = T)

#Conversion to formats:
#for structure analysis
gl2structure(myg, outpath = './',outfile = paste(pref, ".str", sep = ""))

#saving the variant format
myg@other$loc.metrics$position<-myg@position
myg@other$loc.metrics$chromosome<-myg@chromosome
prefix1<-paste("filt_", pref,sep="")
gl2vcf(myg, plink_path = "/bin/", outfile = prefix1, outpath = "./",snp_pos = "position",snp_chr = "chromosome")

prefix2<-paste("filt_", pref, ".vcf",sep="")
system(paste0("sed -i 's/pop1_//g' ", prefix2)) #This will substitute only if the gl object has no pop factor (the default is pop1_)

