# %%
#Script for machine learning step
# %%
from Bio import SeqIO
from Bio import Entrez
from sklearn.cluster import KMeans
from shutil import copyfile
from urllib.request import urlopen
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier as rf #using
from sklearn.model_selection import StratifiedKFold #using
from xgboost import XGBClassifier #using
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
meta=pd.read_csv("../results/ML/filt_biallelic_filtered_classified_meta_K2.csv", index_col=0)
meta

# %%
gen=pd.read_csv("../results/ML/filt_biallelic_filtered.genotype.csv", index_col=0)
gen

# %%
k=5

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
d=split_fun(5, meta=meta, gen=gen)

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
get_cm(k,d,t_xgb,"xgb")

# %%
get_means(d,t_xgb,k)
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
#This is for all other algorithms
def get_forward_snps(i,models,n,dicts):
    folds=list(dicts.keys())
    folds_test=[elem for elem in folds if elem.startswith('train_')]
    folds_train=[elem for elem in folds_test if elem.endswith(format(i))]
    t=SequentialFeatureSelector(list(models.values())[i-1], n_features_to_select=n)
    model=t.fit(dicts[folds_train[0]],dicts[folds_train[1]])
    selected_features=model.get_support()
    return selected_features
def get_forward_snps_parallel(gen,models,n,dicts,k):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures=[]
        for i in range(1,k+1):
            future = executor.submit(get_forward_snps, i, models, n, dicts)
            futures.append(future)
    
    results = [future.result() for future in futures]
    reduced = np.logical_and.reduce(results)
    return reduced

# %% [markdown]
#I need to rank the models based on accuracy and report it
#
#I need to add importance extraction and forward feature selection
# %%
importants=list(t2.values())[4].feature_importances_
snpnames=gen.columns
pd.DataFrame({'Feature':snpnames,'Importance':importants}).sort_values('Importance', ascending=False)
# %%
len(test_x)

# %%
fold_4[0]

# %%
testarray=meta.iloc[fold_1[1]].Classification_K2.to_numpy()
testarray

# %%
for i, (train_index, test_index) in enumerate(skf.split(gen, classes)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
