# %%
#Script for machine learning step
# %%
from Bio import SeqIO
from Bio import Entrez
from sklearn.cluster import KMeans
from shutil import copyfile
from urllib.request import urlopen
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel #Using
from sklearn.ensemble import RandomForestClassifier as rf #using
from sklearn.model_selection import StratifiedKFold #using
from xgboost import XGBClassifier #using
from sklearn.feature_selection import SelectKBest, chi2 #using feature selection
from sklearn.preprocessing import LabelEncoder #using
from sklearn.feature_selection import SelectFromModel #using
from sklearn.neighbors import KNeighborsClassifier as knn #using
from sklearn.naive_bayes import GaussianNB #using
from sklearn import svm #using
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #using
from matplotlib.backends.backend_pdf import PdfPages #using
from sklearn.feature_selection import SequentialFeatureSelector #using
import concurrent.futures #using
import os
import glob
import time
import re
import pickle
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
# %%
#Split validation set
def split_train_test(meta,gen):
    classes=meta.iloc[:,-1].values
    classes=LabelEncoder().fit_transform(classes)
    X_train, X_test, y_train, y_test,indices_train,indices_test =\
    train_test_split(gen.values, classes, gen.index,
                     test_size=0.2,
                     random_state=0,
                     stratify=classes)
    d={}
    d["train_x"]=X_train
    d["test_x"]=X_test
    d["train_y"]=y_train
    d["test_y"]=y_test
    d["train_i"]=indices_train
    d["test_i"]=indices_test
    return d
# %%
#This loop adds information of each train/test in k-fold
def split_fun(k,meta,gen):
    skf=StratifiedKFold(n_splits=k, shuffle=False, random_state=None)
    classes=meta.iloc[:,-1].values
    classes=LabelEncoder().fit_transform(classes)
    test=skf.split(gen, classes)
    d={}
    for i,(train_index, test_index) in enumerate(test,start=1):
        train_x=gen.iloc[train_index].values
        test_x=gen.iloc[test_index].values
        train_y=classes[train_index]
        test_y=classes[test_index]
        d[("train_x"+format(i))]=train_x
        d[("train_y"+format(i))]=train_y
        d[("test_x"+format(i))]=test_x
        d[("test_y"+format(i))]=test_y
    return d

# %%
def process_fold_dt(i, dictionary):
    d={}
    files = np.asarray(list(dictionary.keys()))
    files_in_fold = files[np.char.endswith(files, format(i))]
    train_in_fold = files_in_fold[np.char.startswith(files_in_fold, "train")]
    test_in_fold = files_in_fold[np.char.startswith(files_in_fold, "test")]
    clf = tree.DecisionTreeClassifier()
    clf.fit(dictionary[train_in_fold[0]], dictionary[train_in_fold[1]])
    d[train_in_fold[0]]=clf
    return d
def process_fold_rf(i, dictionary):
    d={}
    files = np.asarray(list(dictionary.keys()))
    files_in_fold = files[np.char.endswith(files, format(i))]
    train_in_fold = files_in_fold[np.char.startswith(files_in_fold, "train")]
    test_in_fold = files_in_fold[np.char.startswith(files_in_fold, "test")]
    clf = rf(n_estimators=300)
    clf.fit(dictionary[train_in_fold[0]], dictionary[train_in_fold[1]])
    d[train_in_fold[0]]=clf
    return d
def process_fold_xgb(i,dictionary):
    d={}
    files = np.asarray(list(dictionary.keys()))
    files_in_fold = files[np.char.endswith(files, format(i))]
    train_in_fold = files_in_fold[np.char.startswith(files_in_fold, "train")]
    test_in_fold = files_in_fold[np.char.startswith(files_in_fold, "test")]
    model = XGBClassifier()
    model.fit(dictionary[train_in_fold[0]], dictionary[train_in_fold[1]], verbose=False)
    d[train_in_fold[0]]=model
    return d

def process_fold_knn(i,dictionary):
    d={}
    files = np.asarray(list(dictionary.keys()))
    files_in_fold = files[np.char.endswith(files, format(i))]
    train_in_fold = files_in_fold[np.char.startswith(files_in_fold, "train")]
    test_in_fold = files_in_fold[np.char.startswith(files_in_fold, "test")]
    model = knn()
    model.fit(dictionary[train_in_fold[0]], dictionary[train_in_fold[1]])
    d[train_in_fold[0]]=model
    return d

def process_fold_naiveb(i,dictionary):
    d={}
    files = np.asarray(list(dictionary.keys()))
    files_in_fold = files[np.char.endswith(files, format(i))]
    train_in_fold = files_in_fold[np.char.startswith(files_in_fold, "train")]
    test_in_fold = files_in_fold[np.char.startswith(files_in_fold, "test")]
    model = GaussianNB()
    model.fit(dictionary[train_in_fold[0]], dictionary[train_in_fold[1]])
    d[train_in_fold[0]]=model
    return d
def process_fold_svm(i,dictionary):
    d={}
    files = np.asarray(list(dictionary.keys()))
    files_in_fold = files[np.char.endswith(files, format(i))]
    train_in_fold = files_in_fold[np.char.startswith(files_in_fold, "train")]
    test_in_fold = files_in_fold[np.char.startswith(files_in_fold, "test")]
    model = svm.SVC(decision_function_shape='ovo')
    model.fit(dictionary[train_in_fold[0]], dictionary[train_in_fold[1]])
    d[train_in_fold[0]]=model
    return d

def get_xgb_model_parallel(dictionary, k):
    models = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures=[]
        for i in range(1,k+1):
            future = executor.submit(process_fold_xgb, i, dictionary)
            futures.append(future)
        
        
        # Retrieve the results
        for future in concurrent.futures.as_completed(futures):
            models.update(future.result())
    
    models=dict(sorted(models.items()))
    return models
    
def get_rf_model_parallel(dictionary, k):
    models = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures=[]
        for i in range(1,k+1):
            future = executor.submit(process_fold_rf, i, dictionary)
            futures.append(future)
        
        
        # Retrieve the results
        for future in concurrent.futures.as_completed(futures):
            models.update(future.result())
    
    models=dict(sorted(models.items()))
    return models
def get_dt_model_parallel(dictionary, k):
    models = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures=[]
        for i in range(1,k+1):
            future = executor.submit(process_fold_dt, i, dictionary)
            futures.append(future)
        
        
        # Retrieve the results
        for future in concurrent.futures.as_completed(futures):
            models.update(future.result())
    
    models=dict(sorted(models.items()))
    return models

def get_knn_model_parallel(dictionary, k):
    models = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures=[]
        for i in range(1,k+1):
            future = executor.submit(process_fold_knn, i, dictionary)
            futures.append(future)
        
        
        # Retrieve the results
        for future in concurrent.futures.as_completed(futures):
            models.update(future.result())
    
    models=dict(sorted(models.items()))
    return models
def get_naiveb_model_parallel(dictionary, k):
    models = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures=[]
        for i in range(1,k+1):
            future = executor.submit(process_fold_naiveb, i, dictionary)
            futures.append(future)
        
        
        # Retrieve the results
        for future in concurrent.futures.as_completed(futures):
            models.update(future.result())
    
    models=dict(sorted(models.items()))
    return models
def get_svm_model_parallel(dictionary, k):
    models = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures=[]
        for i in range(1,k+1):
            future = executor.submit(process_fold_svm, i, dictionary)
            futures.append(future)
        
        
        # Retrieve the results
        for future in concurrent.futures.as_completed(futures):
            models.update(future.result())
    
    models=dict(sorted(models.items()))
    return models

# %%
def save_image(filename): 
    
    # PdfPages is a wrapper around pdf  
    # file so there is no clash and create 
    # files with no error. 
    p = PdfPages(filename) 
      
    # get_fignums Return list of existing  
    # figure numbers 
    fig_nums = plt.get_fignums()   
    figs = [plt.figure(n) for n in fig_nums] 
      
    # iterating over the numbers in list 
    for fig in figs:  
        
        # and saving the files 
        fig.savefig(p, format='pdf')  
      
    # close the object 
    p.close()   

# %%
def get_cm(k,dicts,models,outname):
    files=np.asarray(list(dicts.keys()))
    for i in range(1,k+1):
        files_in_fold=files[np.char.endswith(files,format(i))]
        test_in_fold=files_in_fold[np.char.startswith(files_in_fold,"test")]
    # print(list(t_xgb.values())[i-1].score(d[test_in_fold[0]],d[test_in_fold[1]]))
        predictions = list(models.values())[i-1].predict(dicts[test_in_fold[0]])
        cm = confusion_matrix(dicts[test_in_fold[1]], predictions, labels=list(models.values())[i-1].classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(models.values())[i-1].classes_)
        disp.plot()
    save_image((outname + ".pdf"))

# %%
def get_means(dicts,models, k):
    inlist=list(models.keys())
    folds=list(dicts.keys())
    folds_test=[elem for elem in folds if elem.startswith('test_')]
    folds_train=[elem for elem in inlist if elem.startswith('train_')]
    acc=[]
    for i in range(1,k+1):
        names=[elem for elem in folds_test if elem.endswith(format(i))]
        model=models[folds_train[i-1]]
        sc=model.score(dicts[names[0]],dicts[names[1]])
        acc.append(sc)
    mean=[np.mean(acc)]
    sd=[np.std(acc)]
    df=pd.DataFrame({'Mean':mean,'SD':sd})
    return df
# %%
#get validation means
def get_val_means(meta,gen,models,k):
    inlist=list(models.keys())
    classes=meta.iloc[:,-1].values
    classes=LabelEncoder().fit_transform(classes)
    acc=[]
    for i in range(1,k+1):
        model=models[inlist[i-1]]
        sc=model.score(gen.values,classes)
        acc.append(sc)
    mean=[np.mean(acc)]
    sd=[np.std(acc)]
    df=pd.DataFrame({'Mean':mean,'SD':sd})
    return df
# %%
#This is for treebased algorithms
def get_important_snps(gen,models,k):
    d={}
    for i in range(1,k+1):
        importants=list(models.values())[i-1].feature_importances_
        snpnames=gen.columns
        df=pd.DataFrame({'Feature':snpnames,'Importance':importants}).sort_values('Feature', ascending=False)
        d[("fold_"+format(i))]=df
    concatenated_df=pd.concat(d.values())
    grouped_df=concatenated_df.groupby('Feature')['Importance'].agg(['mean', 'std']).reset_index()
    grouped_df.columns=['Feature', 'Importance', 'SD_folds']
    return grouped_df
# %%
#This is for treebased algorithms
def get_important_snps(estimator,meta,gen,n=None):
    classes=meta.iloc[:,-1].values
    classes=LabelEncoder().fit_transform(classes)
    model=SelectFromModel(estimator, max_features=n).fit(gen.values, classes)
    selected_features=model.get_support()
    return selected_features
# Define function for parallel execution of get_important_snps
def get_important_snps_parallel(estimator,meta,gen, n):
    return get_important_snps(estimator, meta, gen, n)

def fs_tree_models(meta,gen, n):
    estimators = [
        (rf(n_estimators=300), meta, gen, n),
        (XGBClassifier(), meta, gen, n),
        (tree.DecisionTreeClassifier(), meta, gen, n)
    ]
    # Parallel execution of get_important_snps for each estimator
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(lambda args: get_important_snps_parallel(*args), estimators)

    # Retrieve results
    rf_mask, xgb_mask, dt_mask = results
    d={}
    d["rf_mask"]=rf_mask
    d["xgb_mask"]=xgb_mask
    d["dt_mask"]=dt_mask
    return d
# %%
#Testing parallel function for that task
def get_forward_snps(estimator,n,meta,gen):
    classes=meta.iloc[:,-1].values
    classes=LabelEncoder().fit_transform(classes)
    t=SequentialFeatureSelector(estimator, n_features_to_select=n,n_jobs=-1)
    model=t.fit(gen.values,classes)
    selected_features=model.get_support()
    return selected_features
# Define function for parallel execution of get_forward_snps
def parallel_get_forward_snps(estimator, n, meta, gen):
    return get_forward_snps(estimator, n, meta, gen)

# List of tuples containing arguments for get_forward_snps
def fs_non_tree_models(meta_train, gen_train,n="auto"):
    estimators = [
        (knn(), n, meta_train, gen_train),
        (GaussianNB(), n, meta_train, gen_train),
        (svm.SVC(decision_function_shape='ovo'), n, meta_train, gen_train)
    ]

    # Parallel execution of get_forward_snps for each estimator
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(lambda args: parallel_get_forward_snps(*args), estimators)

    # Retrieve results
    knn_mask, nvb_mask, svm_mask = results
    d={}
    d["knn_mask"]=knn_mask
    d["nvb_mask"]=nvb_mask
    d["svm_mask"]=svm_mask
    return d

# %%
#Function to select best plotting mode for hierarchical cluster
def rank_report_linkage(gen,meta):
    methods=["single", "complete","average","weighted","centroid","median","ward"]
    classes=meta.iloc[:,-1].values
    classes=LabelEncoder().fit_transform(classes)
    d={}
    for method in methods:
        linked = linkage(gen, method=method)
        clusters=fcluster(linked,3,criterion="maxclust")
        score=adjusted_rand_score(classes, clusters)
        d[method]=score
    max_value= max(d.values())
    max_keys = [key for key, value in d.items() if value == max_value]
    lut = dict(zip(meta.Classification_K2.unique(), ["#9a0200", "#db5856","#ffc0cb"]))
    plots=[]
    for i in max_keys:
        g_train = sns.clustermap(gen, method=i,row_colors=meta.Classification_K2.map(lut))
        plots.append(g_train)
    return plots
# %% [markdown]
#Running code from the functions above
# %%
#Metadata
meta=pd.read_csv("../results/ML/filt_biallelic_filtered_classified_meta_K2.csv", index_col=0)
meta

# %%
#Genotype data
gen=pd.read_csv("../results/ML/filt_biallelic_filtered.genotype.csv", index_col=0)
gen

# %%
#Number of folds
k=5

# %%
#Number of snps to select
#n=30 - OBS: This is gonna be changed to "auto"
# %%
#Creating partitioned data
split=split_train_test(meta, gen)
meta_val=meta.loc[split["test_i"]]
gen_val=gen.loc[split["test_i"]]
meta_train=meta.loc[split["train_i"]]
gen_train=gen.loc[split["train_i"]]

# %%
#Pre-filtering whole training set with univariate feature selection method
classes=meta_train.iloc[:,-1].values
classes=LabelEncoder().fit_transform(classes)
selector=SelectKBest(chi2, k="all")
selector=selector.fit(gen_train, classes)
masked_features=selector.feature_names_in_[selector.pvalues_<0.05]
gen_train_new=gen_train[masked_features]


# %%
#Function to run wrapped feature selection on non-tree algorithms (forward feature selection)
fs_non_tree=fs_non_tree_models(meta_train,gen_filtered, n=30)
# %%
#Feature selection for non-tree models
knn_mask=fs_non_tree["knn_mask"]
nvb_mask=fs_non_tree["nvb_mask"]
svm_mask=fs_non_tree["svm_mask"]
# %%
#Subsetting variants from the original gen_train
gen_train_knn=gen_train.loc[:,knn_mask]
gen_train_nvb=gen_train.loc[:,nvb_mask]
gen_train_svm=gen_train.loc[:,svm_mask]

# %%
#Generating folds for each subseted data
d_knn=split_fun(k,meta_train,gen_train_knn)
d_nvb=split_fun(k,meta_train,gen_train_nvb)
d_svm=split_fun(k,meta_train,gen_train_svm)

# %%
t_svm=get_svm_model_parallel(d_svm,5)
t_nvb=get_naiveb_model_parallel(d_nvb,5)
t_knn=get_knn_model_parallel(d_knn,5)
# %%
#Get means of the fited models throughout k-folds
m_knn=get_means(d_knn,t_knn,k)
m_nvb=get_means(d_nvb,t_nvb,k)
m_svm=get_means(d_svm,t_svm,k)
# %%
print("knn","\n", m_knn)
print("nvb","\n", m_nvb)
print("svm","\n", m_svm)
# %%
#Now run with validation dataset
gen_val_knn=gen_val.loc[:,knn_mask]
gen_val_nvb=gen_val.loc[:,nvb_mask]
gen_val_svm=gen_val.loc[:,svm_mask]

# %%
#Getting validation means
m_val_knn=get_val_means(meta_val,gen_val_knn,t_knn,k)
m_val_nvb=get_val_means(meta_val,gen_val_nvb,t_nvb,k)
m_val_svm=get_val_means(meta_val,gen_val_svm,t_svm,k)

# %%
print("knn","\n", m_val_knn)
print("nvb","\n", m_val_nvb)
print("svm","\n", m_val_svm)

# %%
#Tree-based models
fs_trees=fs_tree_models(meta_train,gen_train)
# %%
#Feature selection for tree models
xgb_mask=fs_trees["xgb_mask"]
dt_mask=fs_trees["dt_mask"]
rf_mask=fs_trees["rf_mask"]
# %%
#Subsetting variants from the original gen_train
gen_train_xgb=gen_train.loc[:,xgb_mask]
gen_train_dt=gen_train.loc[:,dt_mask]
gen_train_rf=gen_train.loc[:,rf_mask]

# %%
#Generating folds for each subseted data
d_xgb=split_fun(k,meta_train,gen_train_xgb)
d_dt=split_fun(k,meta_train,gen_train_dt)
d_rf=split_fun(k,meta_train,gen_train_rf)

# %%
t_xgb=get_xgb_model_parallel(d_xgb,5)
t_dt=get_dt_model_parallel(d_dt,5)
t_rf=get_rf_model_parallel(d_rf,5)
# %%
#Get means of the fited models throughout k-folds
m_xgb=get_means(d_xgb,t_xgb,k)
m_dt=get_means(d_dt,t_dt,k)
m_rf=get_means(d_rf,t_rf,k)

# %%
print("xgb","\n", m_xgb)
print("dt","\n", m_dt)
print("rf","\n", m_rf)

# %%
#Now run with validation dataset
gen_val_xgb=gen_val.loc[:,xgb_mask]
gen_val_dt=gen_val.loc[:,dt_mask]
gen_val_rf=gen_val.loc[:,rf_mask]

# %%
#Getting validation means
m_val_xgb=get_val_means(meta_val,gen_val_xgb,t_xgb,k)
m_val_dt=get_val_means(meta_val,gen_val_dt,t_dt,k)
m_val_rf=get_val_means(meta_val,gen_val_rf,t_rf,k)

# %%
print("xgb","\n", m_val_xgb)
print("dt","\n", m_val_dt)
print("rf","\n", m_val_rf)
# %%
d=split_fun(5, meta=meta_train, gen=gen_train)
# %%
t_knn=get_knn_model_parallel(d, 5)

# %%
t_nvb=get_naiveb_model_parallel(d,5)

# %%
t_svm=get_svm_model_parallel(d,5)

# %%
#Running with parallelization
t_xgb=get_xgb_model_parallel(d, 5)

# %%
t_rf=get_rf_model_parallel(d,5)

# %%
t_dt=get_dt_model_parallel(d,5)

# %%
print("xgb","\n", get_means(d,t_xgb,k))
print("dt", "\n",get_means(d,t_dt,k))
print("rf", "\n",get_means(d,t_rf,k))
print("nvb","\n",get_means(d,t_nvb,k))
print("knn","\n",get_means(d,t_knn,k))
print("svm","\n",get_means(d,t_svm,k))

# %%
get_cm(k,d,t_xgb,"xgb")

# %%
mask_10=get_forward_snps_parallel(gen,t_xgb,10,d,k)
mask_15=get_forward_snps_parallel(gen,t_xgb,15,d,k)
mask_20=get_forward_snps_parallel(gen,t_xgb,20,d,k)
# %%
importance_df=get_important_snps(gen,t_xgb,k)
gen2=gen.loc[:, mask_10]
d2=split_fun(5, meta=meta, gen=gen2)
t_xgb2=get_xgb_model_parallel(d2, 5)
get_means(d2,t_xgb2,k)
importance_df[importance_df['Feature'].isin(gen2.columns)]
# %%
#Mapeamento para plot:
lut = dict(zip(meta.Classification_K2.unique(), ["#9a0200", "#db5856","#ffc0cb"]))
# %%
g_train = sns.clustermap(DUMMIE, method="ward",row_colors=meta.Classification_K2.map(lut))
