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
from joblib import Parallel, delayed #NOT
from sklearn.feature_selection import SelectFromModel #using
from sklearn.neighbors import KNeighborsClassifier as knn #using
from sklearn.naive_bayes import GaussianNB #using
from sklearn import svm #using
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #using
import concurrent.futures #using
from matplotlib.backends.backend_pdf import PdfPages #using
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
meta["Classification_K2"].value_counts()

# %%
gen=pd.read_csv("../results/ML/filt_biallelic_filtered.genotype.csv", index_col=0)
gen

# %%
meta.index.tolist() == gen.index.tolist()

# %%
k=5

# %%
skf=StratifiedKFold(n_splits=k, shuffle=False, random_state=None)

# %%
classes=meta.iloc[:,-1].values
print(classes)
classes=LabelEncoder().fit_transform(classes)
print(classes)


# %%
skf.get_n_splits(gen,classes)

# %%
test=skf.split(gen, classes)

# %%
#This loop adds information of each train/test in k-fold
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

# %%
files=np.asarray(list(d.keys()))
files


# %%
for i in range(1,k+1):
    files_in_fold=files[np.char.endswith(files,format(i))]
    train_in_fold=files_in_fold[np.char.startswith(files_in_fold,"train")]
    test_in_fold=files_in_fold[np.char.startswith(files_in_fold,"test")]
    
    print("fold"+format(i))
    
    print(d[train_in_fold[0]])
    print("length="+format(len(d[train_in_fold[0]])))
    print(d[train_in_fold[1]])
    print("length="+format(len(d[train_in_fold[1]])))

    print(d[test_in_fold[0]])
    print("length="+format(len(d[test_in_fold[0]])))
    print(d[test_in_fold[1]])
    print("length="+format(len(d[test_in_fold[1]])))
        

# %%
#Getting a model for each fold
def get_xgb_model(d, k):
    files=np.asarray(list(d.keys()))
    models = []
    for i in range(1,k+1):
        files_in_fold=files[np.char.endswith(files,format(i))]
        train_in_fold=files_in_fold[np.char.startswith(files_in_fold,"train")]
        test_in_fold=files_in_fold[np.char.startswith(files_in_fold,"test")]
        
        # instanciar o modelo XGBoost
        model = XGBClassifier()
        # chamar o fit para o modelo
        model.fit(d[train_in_fold[0]], d[train_in_fold[1]], verbose=False)
        # fazer previsões em cima do dataset de teste
        #predictions = model.predict(z_test)
        models.append(model)
    return models

# %%
#Getting a model for each fold
def get_rf_model(d, k):
    files=np.asarray(list(d.keys()))
    models = []
    for i in range(1,k+1):
        files_in_fold=files[np.char.endswith(files,format(i))]
        train_in_fold=files_in_fold[np.char.startswith(files_in_fold,"train")]
        test_in_fold=files_in_fold[np.char.startswith(files_in_fold,"test")]
        
        # instanciar o modelo XGBoost
        model = rf(n_estimators=300)
        # chamar o fit para o modelo
        model.fit(d[train_in_fold[0]], d[train_in_fold[1]])
        # fazer previsões em cima do dataset de teste
        #predictions = model.predict(z_test)
        models.append(model)
    return models

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
# Generate correlation matrix
correlation_matrix = gen.corr()

# Set threshold for correlation
threshold = 0.6  # Example threshold

# Find highly correlated variables
highly_correlated_vars = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            highly_correlated_vars.add(colname)

# Print highly correlated variables
print("Highly correlated variables:", highly_correlated_vars)

# Drop highly correlated variables from DataFrame
df_filtered = gen.drop(columns=highly_correlated_vars)

# Display the filtered DataFrame
print("\nFiltered DataFrame:")
print(df_filtered)

# %%
#Running without parallelization
t1=get_xgb_model(d, 5)

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
importants=list(t2.values())[4].feature_importances_
snpnames=gen.columns
pd.DataFrame({'Feature':snpnames,'Importance':importants}).sort_values('Importance', ascending=False)

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
for i in range(1,k+1):
    files_in_fold=files[np.char.endswith(files,format(i))]
    test_in_fold=files_in_fold[np.char.startswith(files_in_fold,"test")]
   # print(list(t_xgb.values())[i-1].score(d[test_in_fold[0]],d[test_in_fold[1]]))
    predictions = list(t_xgb.values())[i-1].predict(d[test_in_fold[0]])
    cm = confusion_matrix(d[test_in_fold[1]], predictions, labels=list(t_xgb.values())[i-1].classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(t_xgb.values())[i-1].classes_)
    disp.plot()
save_image("xgb.pdf")


# %%
for i in range(1,k+1):
    files_in_fold=files[np.char.endswith(files,format(i))]
    test_in_fold=files_in_fold[np.char.startswith(files_in_fold,"test")]
    print(list(t_rf.values())[i-1].score(d[test_in_fold[0]],d[test_in_fold[1]]))
    predictions = list(t_rf.values())[i-1].predict(d[test_in_fold[0]])
    cm = confusion_matrix(d[test_in_fold[1]], predictions, labels=list(t_rf.values())[i-1].classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(t_rf.values())[i-1].classes_)
    disp.plot()
save_image("rf.pdf")

# %%
for i in range(1,k+1):
    files_in_fold=files[np.char.endswith(files,format(i))]
    test_in_fold=files_in_fold[np.char.startswith(files_in_fold,"test")]
    print(list(t_dt.values())[i-1].score(d[test_in_fold[0]],d[test_in_fold[1]]))
    predictions = list(t_dt.values())[i-1].predict(d[test_in_fold[0]])
    cm = confusion_matrix(d[test_in_fold[1]], predictions, labels=list(t_dt.values())[i-1].classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(t_dt.values())[i-1].classes_)
    disp.plot()
save_image("dt.pdf")

# %%
for i in range(1,k+1):
    files_in_fold=files[np.char.endswith(files,format(i))]
    test_in_fold=files_in_fold[np.char.startswith(files_in_fold,"test")]
    print(list(t_knn.values())[i-1].score(d[test_in_fold[0]],d[test_in_fold[1]]))
    predictions = list(t_knn.values())[i-1].predict(d[test_in_fold[0]])
    cm = confusion_matrix(d[test_in_fold[1]], predictions, labels=list(t_knn.values())[i-1].classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(t_knn.values())[i-1].classes_)
    disp.plot()
save_image("knn.pdf")

# %%
for i in range(1,k+1):
    files_in_fold=files[np.char.endswith(files,format(i))]
    test_in_fold=files_in_fold[np.char.startswith(files_in_fold,"test")]
    print(list(t_nvb.values())[i-1].score(d[test_in_fold[0]],d[test_in_fold[1]]))
    predictions = list(t_nvb.values())[i-1].predict(d[test_in_fold[0]])
    cm = confusion_matrix(d[test_in_fold[1]], predictions, labels=list(t_nvb.values())[i-1].classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(t_nvb.values())[i-1].classes_)
    disp.plot()
save_image("nvb.pdf")

# %%
for i in range(1,k+1):
    files_in_fold=files[np.char.endswith(files,format(i))]
    test_in_fold=files_in_fold[np.char.startswith(files_in_fold,"test")]
    print(list(t_svm.values())[i-1].score(d[test_in_fold[0]],d[test_in_fold[1]]))
    predictions = list(t_svm.values())[i-1].predict(d[test_in_fold[0]])
    cm = confusion_matrix(d[test_in_fold[1]], predictions, labels=list(t_svm.values())[i-1].classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(t_svm.values())[i-1].classes_)
    disp.plot()
save_image("svm.pdf")

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

# %%



