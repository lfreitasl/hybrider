# %%
#Script for machine learning step
# %%
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster #using
from sklearn import tree
from sklearn.model_selection import train_test_split #using
from sklearn.feature_selection import SelectFromModel #Using
from sklearn.ensemble import RandomForestClassifier as rf #using
from sklearn.model_selection import StratifiedKFold #using
from xgboost import XGBClassifier #using
from sklearn.feature_selection import SelectKBest, chi2 #using feature selection
from sklearn.preprocessing import LabelEncoder #using
from sklearn.metrics.cluster import adjusted_rand_score #using
from sklearn.feature_selection import SelectFromModel #using
from sklearn.neighbors import KNeighborsClassifier as knn #using
from sklearn.naive_bayes import GaussianNB #using
from sklearn import svm #using
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #using
from matplotlib.backends.backend_pdf import PdfPages #using
from sklearn.feature_selection import SequentialFeatureSelector #using
import concurrent.futures #using
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    # max_value= max(d.values())
    # max_keys = [key for key, value in d.items() if value == max_value]
    # lut = dict(zip(meta.Classification_K2.unique(), ["#9a0200", "#db5856","#ffc0cb"]))
    # plots={}
    # for i in max_keys:
    #     g_train = sns.clustermap(gen, method=i,row_colors=meta.Classification_K2.map(lut))
    #     plots[i]=g_train
    d=sorted(d.items(), key=lambda x:x[1], reverse=True)
    d=dict(d)
    d=pd.DataFrame(list(d.items()), columns=['Best_grouping_Method', 'Rand_Index'])
    d=d.iloc[[0]]
    return d

# %%
#Pre-filtering whole training set with univariate feature selection method
def broader_selection(df,meta,P):
    classes=meta.iloc[:,-1].values
    classes=LabelEncoder().fit_transform(classes)
    selector=SelectKBest(chi2, k="all")
    selector=selector.fit(df, classes)
    masked_features=selector.feature_names_in_[selector.pvalues_<P]
    gen_train_new=df[masked_features]
    return gen_train_new

# %%
#Removing highly correlated variables
def remove_col_f(df, threshold):
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop features 
    df.drop(to_drop, axis=1, inplace=True)

    print('Removed Columns {}'.format(to_drop))
    return df

# %%
#Function to process dicitonaries generated inside snp_selector function
def df_generator(test_means,val_means,hierarchs):
    test_dfs = []
    validation_dfs = []
    grouping_dfs = []

    for key in test_means:
        test_df = test_means[key].copy()
        test_df['Model'] = key
        test_dfs.append(test_df)
        
        validation_df = val_means[key].copy()
        validation_df['Model'] = key
        validation_dfs.append(validation_df)
        
        grouping_df = hierarchs[key].copy()
        grouping_df['Model'] = key
        grouping_dfs.append(grouping_df)

    # Concatenate the DataFrames
    test_combined_df = pd.concat(test_dfs)
    validation_combined_df = pd.concat(validation_dfs)
    grouping_combined_df = pd.concat(grouping_dfs)

    # Set the Model as the index
    test_combined_df.set_index('Model', inplace=True)
    validation_combined_df.set_index('Model', inplace=True)
    grouping_combined_df.set_index('Model', inplace=True)

    # Rename columns for clarity
    test_combined_df.columns = ['Test_Mean', 'Test_SD']
    validation_combined_df.columns = ['Validation_Mean', 'Validation_SD']

    # Combine all DataFrames into one
    combined_df = pd.concat([test_combined_df, validation_combined_df, grouping_combined_df], axis=1)

    # Create a new columns to display best model based on mean of validation accuracy and RI:
    combined_df['Rank'] = combined_df[['Validation_Mean', 'Rand_Index']].mean(axis=1)
    
    # Sort to report
    combined_df = combined_df.sort_values(by='Rank', ascending=False)

    # Return the resulting DataFrame
    return combined_df
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
n=30

# %%
#P_value for chi2 test for univariate feature selection
p=0.05

# %%
#r² value for among variables correlation filtering.
r=0.8

# %%
def snp_selector (meta, gen, k, p, r, n):
    #Creating partitioned data
    split=split_train_test(meta, gen)
    #Validation set
    meta_val=meta.loc[split["test_i"]]
    gen_val=gen.loc[split["test_i"]]
    #Train-Test set
    meta_train=meta.loc[split["train_i"]]
    gen_train=gen.loc[split["train_i"]]

    #Perform first dimensionality reduction
    gen_train=broader_selection(gen_train,meta_train,p)
    gen_train=remove_col_f(gen_train, r)

    #Subseting the validation set to keep same SNPs
    gen_val=gen_val[gen_train.columns]

    #Function to run wrapped feature selection on non-tree algorithms (forward feature selection)
    fs_non_tree=fs_non_tree_models(meta_train,gen_train, n)
    #Tree-based models
    if n=="auto":
        fs_trees=fs_tree_models(meta_train,gen_train, n=None)
    else:
        fs_trees=fs_tree_models(meta_train,gen_train, n)
    
    #Merging dirs
    models_mask={**fs_non_tree,**fs_trees}
    d={} #Store partitions
    t={} #Store fitted models
    gen_model={} #Store train dataframes for each selected feature
    gen_vals={} #Store validation for each set of selected features
    test_means={} #Store mean for fold models accuracy on test datasets
    val_means={} #Store mean for fold models accuracy on valdiation dataset
    hierarchs={} #Store best hierarch cluster model for each set of model best set of snps
    for key in models_mask:
        model_mask=models_mask[key]
        gen_model[key]=gen_train.loc[:,model_mask]
        d_model=split_fun(k,meta_train,gen_model[key])
        d[key]=d_model
        if key == "knn_mask":
            t_model=get_knn_model_parallel(d[key],k)
        if key == "nvb_mask":
            t_model=get_naiveb_model_parallel(d[key],k)
        if key == "svm_mask":
            t_model=get_svm_model_parallel(d[key],k)
        if key == "rf_mask":
            t_model=get_rf_model_parallel(d[key],k)
        if key == "xgb_mask":
            t_model=get_xgb_model_parallel(d[key],k)
        if key == "dt_mask":
            t_model=get_dt_model_parallel(d[key],k)
        t[key]=t_model
        gen_vals[key]=gen_val.loc[:,model_mask]
        m_test=get_means(d[key],t[key],k)
        m_val=get_val_means(meta_val,gen_vals[key],t[key],k)
        test_means[key]=m_test
        val_means[key]=m_val
        #Test best hierarch model
        best_h=rank_report_linkage(gen[gen_model[key].columns], meta)
        hierarchs[key]=best_h
    res_df=df_generator(test_means,val_means,hierarchs)


# %%
get_cm(k,d,t_xgb,"xgb")
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
