#!/usr/bin/env Rscript

### Automated structure/admixture plots ###

# Load the libraries for this R code 
library(pophelper)
library(grid)
library(gridExtra)
library(reshape2)
library(ggplot2)
library(RColorBrewer)


args = commandArgs(trailingOnly=TRUE)

# Define plotting function no loc
plotnoloc <- function(files, admixture=F, admbestk){
    #Generating number of colors
    getcolours <- colorRampPalette(brewer.pal(8, "Dark2"))
    
    if (admixture) {
        if (admbestk){adm_bestk()}
    }
    #Saving all plots into a variable
    if (length(files)>8){mypal <- getcolours(length(files))}
    else {mypal <- brewer.pal(8, "Dark2")}
    plots<-list()
    for (i in 1:length(files)){
        alg_tmp <- alignK(files[i])
        p_temp  <- plotQ(qlist=mergeQ(alg_tmp), sortind="Cluster1",
                grplabsize=2.5,linesize=0.8,pointsize=3,
                ordergrp=TRUE, returnplot=T,exportplot=F,
                showlegend=T, barbordercolour="#A4A4A4",barbordersize=0.1,
                showyaxis=T, splab=paste("K=",i,sep=""), divcol = "black", divtype = 2,
                divsize=0.6, divalpha=1,
                legendkeysize=10,legendtextsize=10,legendmargin=c(2,2,2,0),
                legendrow=1, showindlab=T, indlabsize=7, useindlab=T,
                splabsize=7, clustercol=mypal)
        plots[[i]]<-p_temp
    }
    return(plots)
}


#Define plot function withloc
plotloc<- function(files, samp_meta, admixture=F, admbestk){
    if (admixture){
        if(admbestk){adm_bestk()}
    }
    if (length(files)>8){mypal <- getcolours(length(files))}
    else {mypal <- brewer.pal(8, "Dark2")}
    plots<-list()
    for (i in 1:length(files)){
        k<-i
        tmp <- as.data.frame(files[i][[1]])
        tmp$K <- k[1]
        tmp$Samples <- rownames(tmp)
        tmp <- melt(tmp, id = c("Samples", "K"))
        names(tmp)[3:4] <- c("Group", "Posterior")
        tmp$Region <- samp_meta$POP
        grp.labs <- paste("K =", k)
        names(grp.labs) <- k
        p3 <- ggplot(tmp, aes(x = Samples, y = Posterior, fill = Group))
        p3 <- p3 + geom_bar(stat = "identity")
        p3 <- p3 + facet_grid(K ~ Region, scales = "free_x", space = "free", 
                            labeller = labeller(K = grp.labs))
        p3 <- p3 + theme_bw()
        p3 <- p3 + ylab("Posterior membership probability")
        p3 <- p3 + theme(legend.position='none')
        p3 <- p3 + scale_fill_manual(values=mypal)
        p3 <- p3 + theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 7))
        plots[[i]]<-p3
    }
        return(plots)
}

#Define function for exporting plots
exp_ks<- function(plots, pref, noloc=F){
    #Plotting each K as svg and pdf
    count = 0
    for (i in (1:length(plots))){
        count= count + 1
        pdf(file= paste("./", pref, "_K",i,"_runs.pdf",sep=""), width = 14, height = 5.81)
        if (noloc){grid.arrange(plots[[i]]$plot[[1]])}
        if (!noloc){grid.arrange(plots[[i]])}
        dev.off()

        svg(filename = paste("./", pref, "_K",count, "_runs.svg",sep=""), width = 14, height = 5.81)
        if (noloc){grid.arrange(plots[[i]]$plot[[1]])}
        if (!noloc){grid.arrange(plots[[i]])}
        dev.off()
    }
}

#Define function for export bestK in admixture
adm_bestk<- function(){
    #saving bestk as pdf
    system("cat log*.out | grep 'CV' | awk -F ' ' '{print $4}' > import")
    df<-read.table("./import")
    system("rm import")
    bestk<-ggplot(df, aes(x = seq_along(V1), y = V1)) +
        geom_point() +
        geom_line() +
        labs(x = "Number of K", y = "Cross Validation") +
        theme_minimal()
    pdf(file= paste("./", "admixture", "_K","_crossvalidation.pdf",sep=""), width = 14, height = 5.81)
    grid.arrange(bestk)
    dev.off()
}

#Define function for building dataframes from metadata with Qvalues for each K
build_meta_qvalues<-function(filepath="./", endpattern, metadata, subpattern, str){
    inds<-metadata$samples
    sfiles <- list.files(path = filepath, pattern= endpattern, full.names = T)
    files <- readQ(files= sfiles)
    if(length(unique(sapply(files,nrow)))==1){
        files <- lapply(files,"rownames<-",inds)
    }
    if (str){
        dfs <- readQ(files= sfiles)
        if(length(unique(sapply(dfs,nrow)))==1){
            dfs <- lapply(dfs,"rownames<-",inds)
        }
        #criando lista de dataframes para structure
        tr<-tabulateQ(dfs)
        sm<-summariseQ(tr)
        k<-nrow(sm)
        rep<-unique(sm[,"runs"])
        if (k>3){
            #EvannoMethodStructure
            evannoMethodStructure(data=sm,exportplot = T,writetable=F, exportpath = getwd(), na.rm=T)
        }
        #Saving all merged Q dataframes into a list
        total_files<-length(dfs)
        files<-list()
        count= 0
        for (i in seq(1,total_files, rep)){
            count = count + 1
            alg_tmp <- alignK(dfs[i:(i+(rep-1))])
            files[count]<-mergeQ(alg_tmp)
        }
    }
    meta_dfs<-list()
    for (i in 1:length(files)){
        temp_df<-merge(files[[i]], metadata, by.x=0, by.y=which(colnames(metadata) == "samples"))
        colnames(temp_df)[1]<-"samples"
        colnames(temp_df)<-gsub("Cluster",subpattern,colnames(temp_df))
        meta_dfs[[i]]<-temp_df
    }
    return(meta_dfs)
}

#Define function for exporting csv with metadata and Q values
export_meta_qvalues<-function(writemeta){
    t_adm<-build_meta_qvalues(filepath = "./",endpattern = "*.Q$",metadata = samp_meta,subpattern = "ADMIX_Cluster",str = F)
    t_str<-build_meta_qvalues(filepath = "./",endpattern = "*_f$",metadata = samp_meta,subpattern = "STR_Cluster",str = T)

    #Creating a list with merged dataframes (meta, admix and str) for each K
    merge_list<-list()
    for (i in (1:length(t_adm))){
    exclude_cols<-colnames(t_adm[[i]])[startsWith(colnames(t_adm[[i]]),"ADMIX_")]
    merge_cols <- setdiff(names(t_adm[[i]]), exclude_cols)
    tm<-merge(t_adm[[i]],t_str[[i]], by=merge_cols)
    merge_list[[i]]<-tm
    }

    #Align K values and give them back to the dataframe
    admcolindex<-character()
    strcolindex<-character()
    aligned_runs<-list()
    for (i in (1:length(merge_list))){
        admcolindex<- append(admcolindex, paste("ADMIX_Cluster", i, sep=""))
        strcolindex<- append(strcolindex, paste("STR_Cluster", i, sep=""))
        slistbyk<- list()
        slistbyk[[1]]<-data.frame(merge_list[[i]][,admcolindex], row.names=merge_list[[i]][,"samples"])
        slistbyk[[2]]<-data.frame(merge_list[[i]][,strcolindex], row.names=merge_list[[i]][,"samples"])
        slistbyk<-alignK(as.qlist(slistbyk))
        if (i==1){
            colnames(slistbyk[[1]])[1]<-"ADMIX_Cluster1"
            colnames(slistbyk[[2]])[1]<-"STR_Cluster1"
        }
        aligned_runs[[i]]<-slistbyk
        }

    for (i in (1:length(aligned_runs))){
        for (j in (1:length(aligned_runs[[i]]))){
            colnames<- colnames(aligned_runs[[i]][[j]])
            merge_list[[i]][colnames]<-aligned_runs[[i]][[j]][colnames]
        }
        if (writemeta){write.csv(merge_list[[i]],file= paste("meta","_str_admix_","K",i,".csv", sep="") ,quote=F,row.names=F)}
    }
    return(merge_list)
}

#Define function for formatting aligned K into plotting functions:
format_slist<-function(metalist,str){
    aligned_runs<-list()
    if (str){pref<-"STR_Cluster"}
    if (!str){pref<-"ADMIX_Cluster"}
    for (i in (1:length(metalist))){
        cols<-metalist[[i]][,startsWith(colnames(metalist[[i]]), prefix = pref)]
        rown<-metalist[[i]]["samples"][[1]]
        df<- data.frame(cols, row.names=rown)
        aligned_runs[[i]]<-df
        }
    return(as.qlist(aligned_runs))
}


#Variables from args
admixture<-as.logical(args[1])
structure<-as.logical(args[2])
meta<-args[3]
usepopinfo<-as.logical(args[4])
writecsvs<-as.logical(args[5])

#Metadata for ind names and loc
samp_meta<-read.csv(meta)

##NOVO FRAMEWORK:
if (writecsvs){
    val<-export_meta_qvalues(writemeta = T)
}
if (!writecsvs){
    val<-export_meta_qvalues(writemeta = F)
}
if (admixture){
    t_adm<-format_slist(val, str=F)
    if (!usepopinfo){
    noloc_adm<-plotnoloc(t_adm, admixture = T, admbestk = T)
    exp_ks(noloc_adm,"ADMIXTURE",noloc=T)
    }
    if (usepopinfo){
    loc_adm<-plotloc(t_adm, samp_meta = samp_meta, admixture = T, admbestk=T)
    exp_ks(loc_adm,"ADMIXTURE",noloc=F)
    }
}
if (structure){
    t_str<-format_slist(val, str=T)
    if (!usepopinfo){
        noloc_str<-plotnoloc(t_str, admixture = F, admbestk = F)
        exp_ks(noloc_str,"STR",noloc=T)
    }
    if (usepopinfo){
        loc_str<-plotloc(t_str, samp_meta = samp_meta, admixture = F, admbestk=F)
        exp_ks(loc_str,"STR",noloc=F)
    }
}
