#!/usr/bin/env python
# %%
#Script for machine learning step
# %%
from scipy.cluster.hierarchy import linkage, fcluster #using
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef #using
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
import argparse
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
    d_cm={}
    for i in range(1,k+1):
        files_in_fold=files[np.char.endswith(files,format(i))]
        test_in_fold=files_in_fold[np.char.startswith(files_in_fold,"test")]
    # print(list(t_xgb.values())[i-1].score(d[test_in_fold[0]],d[test_in_fold[1]]))
        predictions = list(models.values())[i-1].predict(dicts[test_in_fold[0]])
        cm = confusion_matrix(dicts[test_in_fold[1]], predictions, labels=list(models.values())[i-1].classes_)
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(models.values())[i-1].classes_)
    #     disp.plot()
    # save_image((outname + ".pdf"))
        d_cm[list(models.keys()[i-1])]=cm
    return d_cm

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
#Test substitue for val_means, get_means and df_generator
def df_generator(gen_dicts,meta,t,hierarchs=False):
    mt=LabelEncoder().fit_transform(np.array(meta.iloc[:,-1]))
    metrics_list=[]

    for models, folds in t.items():
        for fold, model in folds.items():
            val_t=gen_dicts[models]
            pred=model.predict(val_t)
            accuracy = accuracy_score(mt, pred)
            precision = precision_score(mt, pred, average='macro')  # 'macro' averages the precision of each class
            recall = recall_score(mt, pred, average='macro')        # 'macro' averages the recall of each class
            f1 = f1_score(mt, pred, average='macro')                # 'macro' averages the f1 score of each class
            mcc = matthews_corrcoef(mt, pred)
            metrics_dict = {
                'Model': models,
                'Fold': fold,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'MCC': mcc
            }
            metrics_list.append(metrics_dict)
    
    metrics_df=pd.DataFrame(metrics_list)
    mean_metrics_df = metrics_df.groupby('Model').mean(numeric_only=True).reset_index()
    mean_metrics_df.set_index('Model',inplace=True)
    sorted = mean_metrics_df.sort_values(by='Accuracy', ascending=False)
    sorted.index = sorted.index.str.replace('_mask', '')
    if hierarchs:
        grouping_dfs = []
        for key in hierarchs:
            grouping_df = hierarchs[key].copy()
            grouping_df['Model'] = key
            grouping_dfs.append(grouping_df)
        grouping_combined_df = pd.concat(grouping_dfs)
        grouping_combined_df.set_index('Model', inplace=True)
        sorted_hierarch=grouping_combined_df.sort_values(by='Rand_Index', ascending=False)
        sorted_hierarch.index= sorted_hierarch.index.str.replace('_mask', '')
        return sorted_hierarch
    else:
        return sorted
    
# %%
def writing_fun(gen_all,meta,gen_selected,snp_info,test, val, hierarchs, name="output.txt"):
    best_val=val.index[0]
    best_hierarchical=hierarchs.index[0]
    group_method=hierarchs.loc[best_hierarchical,"Best_grouping_Method"]
    
    snps_hierarch=gen_selected[best_hierarchical+'_mask'].columns
    snps_val=gen_selected[best_val+'_mask'].columns

    table_hierarch=snp_info[snp_info['ID'].isin(snps_hierarch)]
    table_val=snp_info[snp_info['ID'].isin(snps_val)]

    with open(name, 'w') as file:
        file.write('Models metrics for testing sample set:\n')
        file.write(test.to_string(index=True))
        file.write('\n\n')  # Adding a couple of new lines for separation
        file.write('Models metrics for validation sample set:\n')
        file.write(val.to_string(index=True))
        file.write('\n\n Best model: ' + best_val)
        file.write('\n\n')
        file.write('Set of selected variants by '+ best_val + ': \n')
        file.write(table_val.to_string(index=False))
        file.write('\n\n')
        file.write('Model that returned best set of SNPs for hierarchical clustering: ' + best_hierarchical + '\n')
        file.write(hierarchs.to_string(index=True))
        file.write('\n\n')
        file.write('Set of selected variants by ' + best_hierarchical + ': \n')
        file.write(table_hierarch.to_string(index=False))
        file.write('\n\n')
        file.write('This file contains the performance metrics of different models.\n')
        file.write('Each model was trained and tested across different folds.\n')
        file.write('End of report.\n')

    lut = dict(zip(meta.Classification_K2.unique(), ["#9a0200", "#db5856","#ffc0cb"]))
    plot_matrix=gen_all[gen_selected[best_hierarchical+'_mask'].columns]
    plot_h = sns.clustermap(plot_matrix, method=group_method,row_colors=meta.Classification_K2.map(lut))
    plot_h.savefig((best_hierarchical +'_hcluster.jpg'), dpi=400)
    plot_h.savefig((best_hierarchical +'_hcluster.svg'), dpi=400)

# %%
def snp_selector (meta, gen, snp_info, k, p, r, n):
    #Parsing snp_info
    snp_info.drop(['REF','ALT','Alf1','Alf2'], axis=1, inplace=True)
    snp_info.rename(columns={'CHROMOSSOME': 'CHROM','LOC':'ID'}, inplace=True)
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
        #Test best hierarch model
        best_h=rank_report_linkage(gen[gen_model[key].columns], meta)
        hierarchs[key]=best_h
    test_df=df_generator(gen_model,meta_train,t)
    val_df=df_generator(gen_vals,meta_val,t)
    hierarchs_df=df_generator(gen_vals,meta_val,t, hierarchs)
    
    writing_fun(gen,meta,gen_model,snp_info,test_df,val_df,hierarchs_df, name=("SNP_selector_report_"+str(n)+".txt"))

# %%
#Main function
def main():
    parser = argparse.ArgumentParser(description='RNAmining: a deep learning stand-alone and web server tool for sequences coding prediction and RNA functional assignation')
    parser.add_argument('-g','--genotype', help='The filename with the genotype file', required=True)
    parser.add_argument('-m','--metadata', help='The file with metadata information.', required=True)
    parser.add_argument('-i','--snp_info', help='File with each SNP information (Chromossome, position and ID)', required=True)
    parser.add_argument('-k','--k_fold', type=int, help='Number of folds to run crossvalidation on the dataset. Default=5')
    parser.add_argument('-p','--p_value', type=float, help='Number of p_value threshold for initial dimentionality reduction with chi2 test. Default=0.05')
    parser.add_argument('-r','--corr', type=float, help='Number of r2 threshold for initial dimentionality reduction removing highly correlated genotypes. Default=0.8')
    parser.add_argument('-n', '--n_snps', type=int, help='Number of SNPs for algorithm selection. The higher this number the longer is gonna take. Default=auto')
    args = vars(parser.parse_args())
    meta=pd.read_csv(args['metadata'], index_col=0)
    gen=pd.read_csv(args['genotype'], index_col=0)
    snp_info=pd.read_csv(args['snp_info'])

    snp_selector(meta,gen,snp_info,args['k_fold'],args['p_value'],args['corr'],args['n_snps'])
    
# %%
#Run script
if __name__ == "__main__":
    main()