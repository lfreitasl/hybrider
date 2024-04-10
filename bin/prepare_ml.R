#!/usr/bin/env Rscript

### Transformando dados de VCF em tabela para analise de cluster hierarquico ###

### Prep baseado na frequência de alelos dos libidinosus ou de qualquer subamostragem (ver funcao genotypes)
library(vcfR)
library(dplyr)
library(tools)
library(dartR)

# Define function for creating a two or three col matrix (if threecol it will be used in genotype function)
twocol <- function(vcfpath, threecols = T) {
  phy2 <- read.vcfR(
    vcfpath,
    limit = 1e+07,
    nrows = -1,
    skip = 0,
    cols = NULL,
    convertNA = TRUE,
    checkFile = TRUE,
    check_keys = TRUE,
    verbose = TRUE
  )


  SNPs_phy2 <- extract.haps(phy2, mask = FALSE, unphased_as_NA = FALSE, verbose = TRUE)
  SNPs_phy2 <- t(SNPs_phy2)

  phy2tidy <- vcfR2tidy(phy2)
  isolates <- unique(phy2tidy$gt$Indiv)

  # Extraindo rows impares e pares:
  parouimpar <- seq_len(nrow(SNPs_phy2)) %% 2
  data_odd <- SNPs_phy2[parouimpar == 1, ]
  colnames(data_odd) <- as.matrix(paste(colnames(data_odd), "_0", sep = "")) # renomeando colunas para juntar
  rownames(data_odd) <- as.matrix(isolates)

  data_even <- SNPs_phy2[parouimpar == 0, ]
  colnames(data_even) <- as.matrix(paste(colnames(data_even), "_1", sep = ""))
  rownames(data_even) <- as.matrix(isolates)


  # Construindo matriz com duas colunas por locu:

  test <- cbind(data_odd, data_even)
  col.order <- sort(colnames(test))
  test <- test[, col.order]

  if (threecols == F) {
    return(test)
  }

  if (threecols == T) {
    # Adicionar coluna de genotipos (para representação com uma coluna)
    data_genotype <- SNPs_phy2[parouimpar == 0, ]
    colnames(data_genotype) <- as.matrix(paste(colnames(data_genotype), "_2", sep = ""))
    rownames(data_genotype) <- as.matrix(isolates)
    data_genotype[, ] <- NA

    # Construindo matriz com uma colunas por locu:
    test <- cbind(data_odd, data_even, data_genotype)
    col.order <- sort(colnames(test))
    test <- test[, col.order]

    return(test)
  }
}

# Define function that classifies genotype based on allele frequency of one group
genotypes <- function(matriz, df_information, vcfp) {
  # Iterates over dataframe to check allele combination:
  count <- 0
  for (i in seq(1, length(colnames(matriz)), by = 3)) {
    count <- count + 1
    for (j in 1:length(row.names(matriz))) { # Iterate over each sample
      if (!any(is.na(matriz[j, i:(i + 1)]))) { # Only iterates if variant is not NA
        # print(as.character(matriz[j,i:i+1]))
        if (all(as.character(matriz[j, i:(i + 1)]) == df_information[count, "REF"])) { # If both snps equal to ref
          if (df_information[count, "Alf1"] > 0.5) { # Check if ref is major according to alf from selected group
            matriz[j, i + 2] <- 2 # If yes it will be homozygous for dominant allele
          }
          if (df_information[count, "Alf1"] < 0.5) { # Check if ref is minor according to alf from selected group
            matriz[j, i + 2] <- 0 # If yes it will be homozygous for minor allele
          }
        }

        if (all(as.character(matriz[j, i:(i + 1)]) == df_information[count, "ALT"])) { # If both snps equal to alt
          if (df_information[count, "Alf2"] > 0.5) { # Check if alt is major according to alf from selected group
            matriz[j, i + 2] <- 2 # If yes it will be homozygous for dominant allele
          }
          if (df_information[count, "Alf2"] < 0.5) { # Check if alt is minor according to alf from selected group
            matriz[j, i + 2] <- 0 # If yes it will be homozygous for minor allele
          }
        }

        if (any(as.character(matriz[j, i:(i + 1)]) %in% df_information[count, "REF"]) & any(as.character(matriz[j, i:(i + 1)]) %in% df_information[count, "ALT"])) {
          # If both snps are present in both alleles at least one time
          matriz[j, i + 2] <- 1 # Individuals will be heterozygous
        }
        # If the allele frequency of selected group is == 0.5, column will remain with NA values
      }
    }
  }

  matriz <- matriz[, endsWith(colnames(matriz), "_2")]
  colnames(matriz) <- gsub("_2$", "", colnames(matriz))

  return(matriz)
}

# Define function that will return metadata with inferred species based on Q values:
infer_species <- function(samp_meta, method, upper, lower) {
  if (sum(startsWith(colnames(samp_meta), "ADMIX")) != 2) {
    stop("The information for ancestry must be provided for K=2 only")
  }
  for (i in 1:2) {
    upthresh <- upper
    lowthresh <- lower
    str <- paste("ADMIX_Cluster", i, sep = "")
    adm <- paste("STR_Cluster", i, sep = "")
    both <- c(str, adm)
    meanName <- paste("MEAN_Cluster", i, sep = "")
    samp_meta[meanName] <- rowMeans(samp_meta[, both])
  }
  if (method == "mean") {
    samp_meta[samp_meta$MEAN_Cluster1 >= upthresh, "Classification_K2"] <- "Sp1"
    samp_meta[samp_meta$MEAN_Cluster1 <= lowthresh, "Classification_K2"] <- "Sp2"
    samp_meta[samp_meta$MEAN_Cluster1 > lowthresh & samp_meta$MEAN_Cluster1 < upthresh, "Classification_K2"] <- "Hyb"
  }
  if (method == "str") {
    samp_meta[samp_meta$STR_Cluster1 >= upthresh, "Classification_K2"] <- "Sp1"
    samp_meta[samp_meta$STR_Cluster1 <= lowthresh, "Classification_K2"] <- "Sp2"
    samp_meta[samp_meta$STR_Cluster1 > lowthresh & samp_meta$STR_Cluster1 < upthresh, "Classification_K2"] <- "Hyb"
  }
  if (method == "admix") {
    samp_meta[samp_meta$ADMIX_Cluster1 >= upthresh, "Classification_K2"] <- "Sp1"
    samp_meta[samp_meta$ADMIX_Cluster1 <= lowthresh, "Classification_K2"] <- "Sp2"
    samp_meta[samp_meta$ADMIX_Cluster1 > lowthresh & samp_meta$ADMIX_Cluster1 < upthresh, "Classification_K2"] <- "Hyb"
  }
  return(samp_meta)
}

remove_invariable <- function(genotyp) {
  t <- data.frame(row.names = row.names(genotyp))
  for (i in 1:length(colnames(genotyp))) {
    if (length(na.omit(unique(genotyp[, colnames(genotyp)[i]]))) == 2) {
      col <- colnames(genotyp)[i]
      t[col] <- genotyp[, i]
    }
    if (length(na.omit(unique(genotyp[, colnames(genotyp)[i]]))) == 3) {
      col <- colnames(genotyp)[i]
      t[col] <- genotyp[, i]
    }
  }
  return(t)
}

# Define function that will organize and export snpinfo and resample inds based on vector:
report_snpinfo <- function(vcfp, inds, export) {
  # Call vcf from path to gl object
  myg <- gl.read.vcf(vcfp)
  # Resample the interesting individuals
  myg_lib <- gl.keep.ind(myg, inds, recalc = T)
  # Create a dataframe with important information:
  alleles <- do.call(rbind, strsplit(myg_lib$loc.all, "/"))
  df_information <- data.frame(myg_lib$chromosome, myg_lib$position, myg_lib$loc.names, alleles)
  colnames(df_information) <- c("CHROMOSSOME", "POS", "LOC", "REF", "ALT")
  # Calculate AF and add to dataframe information
  AF <- gl.alf(myg_lib)
  df_information$Alf1 <- AF[, 1]
  df_information$Alf2 <- AF[, 2]
  df_information <- df_information[order(df_information$LOC), ] # Sort to match matrix
  if (export) {
    filename <- file_path_sans_ext(basename(vcfp))
    filename <- paste(filename, ".snpinfo.csv", sep = "")
    write.csv(df_information, filename, quote = F, row.names = F)
  }
  return(df_information)
}

args <- commandArgs(trailingOnly = TRUE)


vcfp <- args[1]
metadata <- args[2]
pop <- as.logical(args[3])
whichpop <- args[4]
infer <- as.logical(args[5])
smaller <- as.logical(args[6])
method <- args[7] # Either "mean", "str" or "admix"
upper <- as.numeric(args[8]) # Upper limit of species inference
lower <- as.numeric(args[9]) # Lower limit of species inference
rminvariable <- as.logical(args[10]) # Do you wish to remove invariable genotypes from the matrix?
dropna <- as.logical(args[11]) # Drop all the columns that have NA values


samp_meta <- read.csv(metadata)

# Genotyping based on allele frequency of selected population (if you want to specify a species, please use a prefix before the name of each
# population and pass that to whichpop)
if (pop) {
  inds <- samp_meta[startsWith(samp_meta$POP, whichpop), "samples"]
  matriz <- twocol(vcfp, threecols = T)
  snpinfo <- report_snpinfo(vcfp, inds, T)
  final_matrix <- genotypes(matriz, snpinfo, vcfp)
}

# In case you wanna infer species from data ancestry coefficients
if (infer) {
  samp_meta <- infer_species(samp_meta, method, upper, lower)
  filename <- file_path_sans_ext(basename(vcfp))
  filename <- paste(filename, "_classified_meta_K2.csv", sep = "")
  write.csv(samp_meta, filename, quote = FALSE, row.names=FALSE)
  # In case you wanna take allele frequency to genotype based on the smallest group on the dataset
  if (smaller) {
    total_sp1 <- sum(samp_meta$Classification_K2 == "Sp1")
    total_sp2 <- sum(samp_meta$Classification_K2 == "Sp2")
    numberofeach <- c(total_sp1, total_sp2)
    smallgroupindex <- which.min(numberofeach)
    if (smallgroupindex == 1) {
      inds <- samp_meta[startsWith(samp_meta$Classification_K2, "Sp1"), "samples"]
    }
    if (smallgroupindex == 2) {
      inds <- samp_meta[startsWith(samp_meta$Classification_K2, "Sp2"), "samples"]
    }
  }
  # This will take on default all the samples with Sp1 label
  if (!smaller) {
    inds <- samp_meta[startsWith(samp_meta$POP, "Sp1"), "samples"]
  }
  matriz <- twocol(vcfp, threecols = T)
  snpinfo <- report_snpinfo(vcfp, inds, T)
  final_matrix <- genotypes(matriz, snpinfo, vcfp)
}

if (rminvariable) {
  final_matrix <- remove_invariable(final_matrix)
}
if (dropna) {
  final_matrix <- final_matrix[, colSums(is.na(final_matrix)) == 0]
}

filename <- file_path_sans_ext(basename(vcfp))
filename <- paste(filename, ".genotype.csv", sep = "")
write.csv(final_matrix, filename, quote = FALSE, row.names= TRUE)
