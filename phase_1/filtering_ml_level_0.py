from sklearn.cluster import DBSCAN,KMeans,AgglomerativeClustering,compute_optics_graph,cluster_optics_xi,SpectralClustering,Birch,OPTICS
from sklearn.svm import SVC,NuSVC,OneClassSVM,SVR,NuSVR
import pandas as pd
import numpy as np
import datetime
import sklearn
from sklearn.preprocessing import MinMaxScaler,StandardScaler, Normalizer, normalize,minmax_scale
from sklearn.isotonic import IsotonicRegression
from sklearn.svm import LinearSVR, LinearSVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, KernelDensity
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.naive_bayes import CategoricalNB,BernoulliNB,ComplementNB,GaussianNB,MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression, ElasticNet, BayesianRidge, HuberRegressor, PoissonRegressor, PassiveAggressiveRegressor
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, KernelDensity, KDTree
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from scipy.spatial.distance import sqeuclidean,jaccard,canberra,cdist,euclidean
from statistics import median,median_low,median_high,geometric_mean,harmonic_mean,quantiles
from scipy import stats as st
import matplotlib.pyplot as plt
import shap
import torch
import gc
import os
currentTime=str(datetime.datetime.now()).replace(':','_')
device=torch.device("mps")
torch.cuda.empty_cache()
scaler =StandardScaler()
#gc.collect()
#from imblearn.metrics import classification_report_imbalanced
#from imblearn.pipeline import make_pipeline as make_pipeline_imb
def dataLoading(data_dir):
    global cols_erh,cols_rs
    df_orig = pd.read_csv(data_dir)
    print(df_orig.columns)
    df_orig.fillna(0, inplace=True)
    
    columns = df_orig.columns
    cols_rs=[]
    cols_erh= []
    for col in columns:
        if '.rs' in col:
            #print(col)
            cols_rs.append(str(col))
        elif '.erh' in col:
            cols_erh.append(str(col))
    #print(cols_rs)
    
    df_orig ['S1_COUNT']= (df_orig[cols_rs]>0).sum(axis=1)
    
    df_orig ['Richness_COUNT']= (df_orig[cols_erh]>0).sum(axis=1)

    df_orig['S1_STDEV'] = df_orig[cols_rs].std(axis=1)
    df_orig['Richness_STDEV'] = df_orig[cols_erh].std(axis=1)
    df_orig=df_orig[(df_orig['S1_SUM']>0)&(df_orig['Richness_SUM']>0)&(df_orig['S1_COUNT']>0)&(df_orig['Richness_COUNT']>0)&(df_orig['S1_STDEV']>0)&(df_orig['Richness_STDEV']>0)]
    df_orig['S1_ind']  = df_orig['S1_SUM']/df_orig['S1_STDEV']
    df_orig['Richness_ind']  = df_orig['Richness_SUM']/df_orig['Richness_STDEV']
    df_orig['S1_Richness_efficiency']= df_orig['Richness_SUM']/df_orig['S1_SUM']
    df_orig['S1_Richness_balance']=abs(df_orig['Richness_ind']/df_orig['S1_ind'].apply(np.log))
    df_orig = df_orig[['CodeA','CodeB','CodeC','Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT','S1_ind','Richness_ind','S1_Richness_balance','S1_Richness_efficiency']]
    print(df_orig [(df_orig ['CodeC']==192) & (df_orig ['CodeB']==247) & (df_orig ['CodeA']==169)])
    print(df_orig [(df_orig ['CodeC']==192) & (df_orig ['CodeB']==245) & (df_orig ['CodeA']==191)])
    print(df_orig [(df_orig ['CodeC']==192) & (df_orig ['CodeB']==245) & (df_orig ['CodeA']==235)])
    print(df_orig [(df_orig ['CodeC']==192) & (df_orig ['CodeB']==207) & (df_orig ['CodeA']==134)])
    print(df_orig [(df_orig ['CodeC']==192) & (df_orig ['CodeB']==152) & (df_orig ['CodeA']==173)])
    print(df_orig [(df_orig ['CodeC']==192) & (df_orig ['CodeB']==247) & (df_orig ['CodeA']==174)])
    print(df_orig [(df_orig ['CodeC']==192) & (df_orig ['CodeB']==137) & (df_orig ['CodeA']==233)])
    print(df_orig [(df_orig ['CodeC']==1) & (df_orig ['CodeB']==17) & (df_orig ['CodeA']==71)])
    print(df_orig [(df_orig ['CodeC']==260) & (df_orig ['CodeB']==26) & (df_orig ['CodeA']==22)])
    print(df_orig [(df_orig ['CodeC']==192) & (df_orig ['CodeB']==194) & (df_orig ['CodeA']==202)])
    print(df_orig [(df_orig ['CodeC']==192) & (df_orig ['CodeB']==194) & (df_orig ['CodeA']==52)])
    print(df_orig [(df_orig ['CodeC']==192) & (df_orig ['CodeB']==234) & (df_orig ['CodeA']==173)])
    print(df_orig [(df_orig ['CodeC']==192) & (df_orig ['CodeB']==245) & (df_orig ['CodeA']==156)])


    return df_orig
class preprocess():

    def dataNormalize(df):


        
        print(df.columns)
        cols = ['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']
        for col in cols:
            df[col+'_orig']=df[col]
        df_ = df[cols]

        df_normalized  = pd.DataFrame(scaler.fit_transform(df_.values),columns=df_.columns,index=df_.index)
        print(df_normalized.columns)
        df = df.drop(cols,axis=1)
        df_normalized = pd.concat((df,df_normalized),axis=1)






        print(df_normalized [(df_normalized['CodeC']==192) & (df_normalized ['CodeB']==247) & (df_normalized ['CodeA']==169)])
        print(df_normalized [(df_normalized ['CodeC']==192) & (df_normalized ['CodeB']==245) & (df_normalized ['CodeA']==191)])
        print(df_normalized [(df_normalized ['CodeC']==192) & (df_normalized ['CodeB']==245) & (df_normalized ['CodeA']==235)])
        print(df_normalized [(df_normalized ['CodeC']==192) & (df_normalized ['CodeB']==207) & (df_normalized ['CodeA']==134)])
        print(df_normalized [(df_normalized ['CodeC']==192) & (df_normalized ['CodeB']==152) & (df_normalized ['CodeA']==173)])
        print(df_normalized [(df_normalized ['CodeC']==192) & (df_normalized ['CodeB']==247) & (df_normalized ['CodeA']==174)])
        print(df_normalized [(df_normalized ['CodeC']==192) & (df_normalized ['CodeB']==137) & (df_normalized ['CodeA']==233)])
        print(df_normalized [(df_normalized ['CodeC']==1) & (df_normalized ['CodeB']==17) & (df_normalized ['CodeA']==71)])
        print(df_normalized [(df_normalized ['CodeC']==192) & (df_normalized ['CodeB']==194) & (df_normalized ['CodeA']==202)])
        print(df_normalized [(df_normalized ['CodeC']==192) & (df_normalized ['CodeB']==194) & (df_normalized ['CodeA']==52)])
        print(df_normalized [(df_normalized ['CodeC']==192) & (df_normalized ['CodeB']==234) & (df_normalized ['CodeA']==173)])
        print(df_normalized [(df_normalized ['CodeC']==192) & (df_normalized ['CodeB']==245) & (df_normalized ['CodeA']==156)])



        return df_normalized
    def dataPreprocess(df):
        df=df[(df['S1_SUM']>0)&(df['Richness_SUM']>0)&(df['S1_COUNT']>0)&(df['Richness_COUNT']>0)]



        print(df [(df ['CodeC']==192) & (df ['CodeB']==247) & (df ['CodeA']==169)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==191)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==235)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==207) & (df ['CodeA']==134)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==152) & (df ['CodeA']==173)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==247) & (df ['CodeA']==174)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==137) & (df ['CodeA']==233)])
        print(df [(df ['CodeC']==1) & (df ['CodeB']==17) & (df ['CodeA']==71)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==194) & (df ['CodeA']==202)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==194) & (df ['CodeA']==52)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==234) & (df ['CodeA']==173)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==156)])
        scope_count = int(len(df)*0.1)
        df=df.nlargest(scope_count , ['S1_COUNT','Richness_COUNT'])

        print('-------------------------s1 and Richness count----------------------')
        print(len(df))

        print(df [(df ['CodeC']==192) & (df ['CodeB']==247) & (df ['CodeA']==169)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==191)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==235)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==207) & (df ['CodeA']==134)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==152) & (df ['CodeA']==173)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==247) & (df ['CodeA']==174)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==137) & (df ['CodeA']==233)])
        print(df [(df ['CodeC']==1) & (df ['CodeB']==17) & (df ['CodeA']==71)])
        print(df [(df ['CodeC']==260) & (df ['CodeB']==26) & (df ['CodeA']==22)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==194) & (df ['CodeA']==202)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==194) & (df ['CodeA']==52)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==234) & (df ['CodeA']==173)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==156)])





        scope_balance = int(len(df)*0.4)

        df=df.nlargest(scope_balance , ['S1_ind','Richness_ind'])
        print('-------------------------s1 and Richness balance----------------------')
        print(len(df))

        print(df [(df ['CodeC']==192) & (df ['CodeB']==247) & (df ['CodeA']==169)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==191)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==235)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==207) & (df ['CodeA']==134)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==152) & (df ['CodeA']==173)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==247) & (df ['CodeA']==174)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==137) & (df ['CodeA']==233)])
        print(df [(df ['CodeC']==1) & (df ['CodeB']==17) & (df ['CodeA']==71)])
        print(df [(df ['CodeC']==260) & (df ['CodeB']==26) & (df ['CodeA']==22)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==194) & (df ['CodeA']==202)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==194) & (df ['CodeA']==52)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==234) & (df ['CodeA']==173)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==156)])
        #print(scope_efficiency)







        scope_balance_ind = int(len(df)*0.9)

        df=df.nlargest(scope_balance_ind , 'S1_Richness_balance')
        print('-------------------------s1 and Richness balance----------------------')
        print(len(df))

        print(df [(df ['CodeC']==192) & (df ['CodeB']==247) & (df ['CodeA']==169)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==191)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==235)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==207) & (df ['CodeA']==134)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==152) & (df ['CodeA']==173)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==247) & (df ['CodeA']==174)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==137) & (df ['CodeA']==233)])
        print(df [(df ['CodeC']==1) & (df ['CodeB']==17) & (df ['CodeA']==71)])
        print(df [(df ['CodeC']==260) & (df ['CodeB']==26) & (df ['CodeA']==22)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==194) & (df ['CodeA']==202)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==194) & (df ['CodeA']==52)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==234) & (df ['CodeA']==173)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==156)])
        scope_total_Richness = int(len(df)*0.4)

        #print(scope_efficiency)
        df=df.nlargest(scope_total_Richness , ['Richness_SUM','S1_SUM'])
        print('-------------------------total Richness----------------------')
        print(len(df))

        print(df [(df ['CodeC']==192) & (df ['CodeB']==247) & (df ['CodeA']==169)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==191)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==235)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==207) & (df ['CodeA']==134)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==152) & (df ['CodeA']==173)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==247) & (df ['CodeA']==174)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==137) & (df ['CodeA']==233)])
        print(df [(df ['CodeC']==1) & (df ['CodeB']==17) & (df ['CodeA']==71)])
        print(df [(df ['CodeC']==260) & (df ['CodeB']==26) & (df ['CodeA']==22)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==194) & (df ['CodeA']==202)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==194) & (df ['CodeA']==52)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==234) & (df ['CodeA']==173)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==156)])
    

        df=df[(df['S1_ind']>1)&(df['Richness_ind']>1)]

        print(len(df))
        print(df [(df ['CodeC']==192) & (df ['CodeB']==247) & (df ['CodeA']==169)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==191)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==235)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==207) & (df ['CodeA']==134)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==152) & (df ['CodeA']==173)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==247) & (df ['CodeA']==174)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==137) & (df ['CodeA']==233)])
        print(df [(df ['CodeC']==1) & (df ['CodeB']==17) & (df ['CodeA']==71)])
        print(df [(df ['CodeC']==260) & (df ['CodeB']==26) & (df ['CodeA']==22)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==194) & (df ['CodeA']==202)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==194) & (df ['CodeA']==52)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==234) & (df ['CodeA']==173)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==156)])



        return df

    def outlierFiltering(x,df,round):        
        gc.collect()
        
        n = round

        kde=KernelDensity(kernel='gaussian').fit(x)
        dens=kde.score_samples(x)
        #print(dens)            

        average_dens=np.mean(dens)
        
                
        print('average density is: ', average_dens)          
            
        if abs(average_dens)<2:
            min_samples_=2
        else:
            min_samples_=int(abs(average_dens))
        print("min_samples are: ", min_samples_)
        
        x_ = np.array(x)
        dist=[]
        for i in range (len(x)):

            dist_pair=cdist(np.array(np.reshape(x[i],(1,len(x[i])))),x_,'euclidean')
            average_dist_= np.mean(dist_pair)

            dist.append(average_dist_)
            #print(len(dist))
        print(len(dist))
        average_dist=max(dist)
        print('average distance is: ', average_dist)
        eps_=average_dist/(2*min_samples_)

        print("eps is: ", eps_)
            
        
        print("min_samples are: ", min_samples_)
        
        
        model_classification = DBSCAN(eps=eps_,min_samples=min_samples_)

        y_predict = model_classification.fit_predict(x)
        df['class'] = y_predict
        print(df [(df ['CodeC']==192) & (df ['CodeB']==247) & (df ['CodeA']==169)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==191)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==235)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==207) & (df ['CodeA']==134)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==152) & (df ['CodeA']==173)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==247) & (df ['CodeA']==174)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==137) & (df ['CodeA']==233)])
        print(df [(df ['CodeC']==1) & (df ['CodeB']==17) & (df ['CodeA']==71)])
        print(df [(df ['CodeC']==260) & (df ['CodeB']==26) & (df ['CodeA']==22)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==194) & (df ['CodeA']==202)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==194) & (df ['CodeA']==52)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==234) & (df ['CodeA']==173)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==156)])
        df_filtered=df[df['class']!=-1]
        print(df[df['class']==-1])
        print(len(df_filtered))
        '''
        with open (temp_dir+'eps_minSamples_'+str(n)+'.log','w') as f:
            f.write(str(eps_)+','+str(min_samples_))
            f.close()
        '''
        return df_filtered

def classification(df_filtered):
    print(len(df_filtered))
    df_result = df_filtered
    x = np.array(df_result[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']])
    y_predict,model_name,score_of_determination = ClusteringAnalysis.oneclassSVM(x)
    #y_predict,model_name  = ClusteringAnalysis.kmeans(df_filtered)
    #y_predict,model_name = ClusteringAnalysis.oneclassSVM(df_filtered)
    #y_predict,model_name = ClusteringAnalysis.oneclassSVM_simple(df)
    #y_predict,model_name  = ClusteringAnalysis.spectral(df_filtered)

    #y_predict,model_name  = ClusteringAnalysis.birch(df_filtered)
    #y_predict,model_name  = ClusteringAnalysis.agglomerativeClustering(df_filtered)
    #y_predict,model_name  = ClusteringAnalysis.opticsClustering(df_filtered)
    df_result['Score_OSVM'] = score_of_determination
    df_result['class'] = y_predict
    df_result = df_result.sort_values(by =['Score_OSVM'],ascending=True)
    df_result['SCORE_RANK_OSVM']=df_result['Score_OSVM'].rank(ascending=True)

    print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==247) & (df_result ['CodeA']==169)])
    print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==245) & (df_result ['CodeA']==191)])
    print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==245) & (df_result ['CodeA']==235)])
    print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==207) & (df_result ['CodeA']==134)])
    print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==152) & (df_result ['CodeA']==173)])
    print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==247) & (df_result ['CodeA']==174)])
    print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==137) & (df_result ['CodeA']==233)])
    print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==194) & (df_result ['CodeA']==202)])
    print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==194) & (df_result ['CodeA']==52)])
    print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==234) & (df_result ['CodeA']==173)])
    print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==245) & (df_result ['CodeA']==156)])
    print(df_result [(df_result ['CodeC']==1) & (df_result ['CodeB']==17) & (df_result ['CodeA']==71)])
    print(df_result [(df_result ['CodeC']==260) & (df_result ['CodeB']==26) & (df_result ['CodeA']==22)])
    result_name='output_'+model_name+'.csv'
    df_result.to_csv(output_dir+result_name)
    return y_predict,model_name,x,df_result 

class similarityAnalysis():
    def erhAnalysis(erh_dir,df_filtered):
        print(len(df_filtered))
        df_merge = df_filtered

        array_1 = [[1,3],[4,6],[7,9],[10,12],[13,15],[16,18]]
        array_2 = [[2,3],[5,6],[8,9],[11,12],[14,15],[17,18]]
        col_1355=[]
        col_insr=[]
        col_1361=[]    
        for r,d,f in os.walk(erh_dir):
            print(r)
            for m in range(len(array_1)):
                num_1355 =array_1[m][0]
                num_1361 =array_2[m][0]
                num_insr =array_1[m][1]
                for f_ in f:
                    if num_1355 == int(f_.split("-")[0]):
                        #print('pY1355 col file is: ',f_)
                        col_1355.append(f_)
                    elif num_1361 == int(f_.split("-")[0]):
                        #print('pY1361 col file is: ',f_)
                        col_1361.append(f_)
                    elif num_insr == int(f_.split("-")[0]):
                        #print('INSR col file is: ',f_)
                        col_insr.append(f_)
            for i in range(len(f)):
                f_ = f[i]
                if f_.split(".")[1]=="erh":
                    if "pY1355" in f_.split(".")[0]:
                        print('pY1355 file is: ',f_)

                        df_1355 = pd.read_csv(r+f_,names=['CodeA','CodeB','CodeC','S1','Richness'] )
                        df_1355 = df_1355.sort_values(by=['CodeA','CodeB','CodeC'])
                        col_name = str(f_)
                        df_1355[col_name] = df_1355['Richness']
                        df_1355  = df_1355.drop("Richness",axis=1)
                        df_1355[col_name+'_S1'] = df_1355['S1']
                        df_1355 = df_1355.drop("S1",axis=1)
                        df_merge = df_merge.merge(df_1355,how='left',on=['CodeA','CodeB','CodeC'])
                    elif "pY1361" in f_.split(".")[0]:
                        print('pY1361 file is: ',f_)
                        df_1361 = pd.read_csv(r+f_,names=['CodeA','CodeB','CodeC','S1','Richness'])
                        df_1361 = df_1361.sort_values(by=['CodeA','CodeB','CodeC'])
                        col_name = str(f_)
                        df_1361[col_name] = df_1361['Richness']
                        df_1361  = df_1361.drop("Richness",axis=1)
                        df_1361[col_name+'_S1'] = df_1361['S1']
                        df_1361 = df_1361.drop("S1",axis=1)
                        df_merge = df_merge.merge(df_1361,how='left',on=['CodeA','CodeB','CodeC'])
                    elif "INSR" in f_.split(".")[0]:
                        print('INSR file is: ',f_)
                        df_ins = pd.read_csv(r+f_,names=['CodeA','CodeB','CodeC','S1','Richness'])
                        df_ins = df_ins.sort_values(by=['CodeA','CodeB','CodeC'])
                        print(df_ins [(df_ins ['CodeC']==42) & (df_ins ['CodeB']==703) & (df_ins ['CodeA']==327)])
                        col_name = str(f_)
                        df_ins[col_name] = df_ins['Richness']
                        df_ins = df_ins.drop("Richness",axis=1)
                        df_ins[col_name+'_S1'] = df_ins['S1']
                        df_ins = df_ins.drop("S1",axis=1)
                        df_merge = df_merge.merge(df_ins,how='left',on=['CodeA','CodeB','CodeC'])
                        print(df_merge [(df_merge ['CodeC']==42) & (df_merge ['CodeB']==703) & (df_merge ['CodeA']==327)])

        df_merge.fillna(0,inplace=True)
        print(df_merge [(df_merge ['CodeC']==1457) & (df_merge ['CodeB']==210) & (df_merge ['CodeA']==104)])
        df_col = pd.DataFrame()
        df_col['col_1355']=col_1355
        df_col['col_1361']=col_1361
        df_col['col_insr']=col_insr
        print(df_col)
        
        df_merge['performance_ind_0_total']=0
        df_merge['richness_1355_0']=0
        df_merge['richness_1355_1']=0
        df_merge['performance_ind_1_total']=0
        df_merge['richness_1361_0']=0
        df_merge['richness_1361_1']=0
        df_merge['richness_insr_1355_0']=0
        df_merge['richness_insr_1355_1']=0
        df_merge['richness_insr_1361_0']=0
        df_merge['richness_insr_1361_1']=0
        df_merge['s1_1355_0']=0
        df_merge['s1_1355_1']=0
        df_merge['s1_1361_0']=0
        df_merge['s1_1361_1']=0
        df_merge['s1_insr_1355_0']=0
        df_merge['s1_insr_1355_1']=0
        df_merge['s1_insr_1361_0']=0
        df_merge['s1_insr_1361_1']=0

        for i in range(len(df_merge)):
            richiness_1355_1 = 0.00
            richiness_1355_0 = 0.00
            richiness_insr_1355_1 = 0.00
            richiness_insr_1355_0 = 0.00
            richiness_1361_1 = 0.00
            richiness_1361_0 = 0.00
            richiness_insr_1361_1 = 0.00
            richiness_insr_1361_0 = 0.00
            s1_1355_1 = 0.00
            s1_1355_0 = 0.00
            s1_insr_1355_1 = 0.00
            s1_insr_1355_0 = 0.00
            s1_1361_1 = 0.00
            s1_1361_0 = 0.00
            s1_insr_1361_1 = 0.00
            s1_insr_1361_0 = 0.00
            performance_total_1355 = 0
            performance_total_1361 = 0
            for j in range(len(df_col)):
                col_1355_ = df_col['col_1355'].values[j]
                col_1361_ = df_col['col_1361'].values[j]
                col_insr_ = df_col['col_insr'].values[j]
                col_1355_S1 = df_col['col_1355'].values[j]+'_S1'
                col_1361_S1 = df_col['col_1361'].values[j]+'_S1'
                col_insr_S1 = df_col['col_insr'].values[j]+'_S1'
                #print(col_1355_,col_insr_)   
                if df_merge[col_1355_].values[i]>=df_merge[col_insr_].values[i]:
                    richiness_1355_1+=df_merge[col_1355_].values[i]
                    richiness_insr_1355_1+=df_merge[col_insr_].values[i]
                    s1_1355_1+=df_merge[col_1355_S1].values[i]
                    s1_insr_1355_1+=df_merge[col_insr_S1].values[i]
                    performance_total_1355+=1
                else:
                    richiness_1355_0+=df_merge[col_1355_].values[i]
                    performance_total_1355+=0
                    richiness_insr_1355_0+=df_merge[col_insr_].values[i]
                    s1_1355_0+=df_merge[col_1355_S1].values[i]
                    s1_insr_1355_0+=df_merge[col_insr_S1].values[i]
                #print(col_1361_,col_insr_)   
                if df_merge[col_1361_].values[i]>=df_merge[col_insr_].values[i]:
                    richiness_1361_1+=df_merge[col_1361_].values[i]
                    richiness_insr_1361_1+=df_merge[col_insr_].values[i]
                    performance_total_1361+=1
                    s1_1361_1+=df_merge[col_1361_S1].values[i]
                    s1_insr_1361_1+=df_merge[col_insr_S1].values[i]
                else:
                    richiness_1361_0+=df_merge[col_1361_].values[i]
                    performance_total_1361+=0
                    richiness_insr_1361_0+=df_merge[col_insr_].values[i]
                    s1_1361_0+=df_merge[col_1361_S1].values[i]
                    s1_insr_1361_0+=df_merge[col_insr_S1].values[i]
            df_merge['performance_ind_0_total'].values[i]=performance_total_1355
            df_merge['richness_1355_0'].values[i]=richiness_1355_0
            df_merge['richness_1355_1'].values[i]=richiness_1355_1
            df_merge['performance_ind_1_total'].values[i]=performance_total_1361
            df_merge['richness_1361_0'].values[i]=richiness_1361_0
            df_merge['richness_1361_1'].values[i]=richiness_1361_1
            df_merge['richness_insr_1355_0'].values[i]=richiness_insr_1355_0
            df_merge['richness_insr_1355_1'].values[i]=richiness_insr_1355_1
            df_merge['richness_insr_1361_0'].values[i]=richiness_insr_1361_0
            df_merge['richness_insr_1361_1'].values[i]=richiness_insr_1361_1
            df_merge['s1_1355_0'].values[i]=s1_1355_0
            df_merge['s1_1355_1'].values[i]=s1_1355_1
            df_merge['s1_1361_0'].values[i]=s1_1361_0
            df_merge['s1_1361_1'].values[i]=s1_1361_1
            df_merge['s1_insr_1355_0'].values[i]=s1_insr_1355_0
            df_merge['s1_insr_1355_1'].values[i]=s1_insr_1355_1
            df_merge['s1_insr_1361_0'].values[i]=s1_insr_1361_0
            df_merge['s1_insr_1361_1'].values[i]=s1_insr_1361_1
        print(df_merge [(df_merge ['CodeC']==1457) & (df_merge ['CodeB']==210) & (df_merge ['CodeA']==104)])
        df_merge['total_pY1355_mixed']=0
        df_merge['total_pY1361_mixed']=0
        df_merge['total_insr_pY1355_mixed']=0
        df_merge['total_insr_pY1361_mixed']=0
        df_merge['s1_pY1355_mixed']=0
        df_merge['s1_pY1361_mixed']=0
        df_merge['s1_insr_pY1355_mixed']=0
        df_merge['s1_insr_pY1361_mixed']=0
        for i in range(len(df_merge)):
            if df_merge['performance_ind_0_total'].values[i]>=3:
                df_merge['total_pY1355_mixed'].values[i]=df_merge['richness_1355_1'].values[i]
                df_merge['total_insr_pY1355_mixed'].values[i]=df_merge['richness_insr_1355_1'].values[i]
                df_merge['s1_pY1355_mixed'].values[i]=df_merge['s1_1355_1'].values[i]
                df_merge['s1_insr_pY1355_mixed'].values[i]=df_merge['s1_insr_1355_1'].values[i]
            else:
                df_merge['total_pY1355_mixed'].values[i]=df_merge['richness_1355_0'].values[i]
                df_merge['total_insr_pY1355_mixed'].values[i]=df_merge['richness_insr_1355_0'].values[i]
                df_merge['s1_pY1355_mixed'].values[i]=df_merge['s1_1355_0'].values[i]
                df_merge['s1_insr_pY1355_mixed'].values[i]=df_merge['s1_insr_1355_0'].values[i]
        for i in range(len(df_merge)):
            if df_merge['performance_ind_1_total'].values[i]>=3:
                df_merge['total_pY1361_mixed'].values[i]=df_merge['richness_1361_1'].values[i]
                df_merge['total_insr_pY1361_mixed'].values[i]=df_merge['richness_insr_1361_1'].values[i]
                df_merge['s1_pY1361_mixed'].values[i]=df_merge['s1_1361_1'].values[i]
                df_merge['s1_insr_pY1361_mixed'].values[i]=df_merge['s1_insr_1361_1'].values[i]
            else:
                df_merge['total_pY1361_mixed'].values[i]=df_merge['richness_1361_0'].values[i]
                df_merge['total_insr_pY1361_mixed'].values[i]=df_merge['richness_insr_1361_0'].values[i]
                df_merge['s1_pY1361_mixed'].values[i]=df_merge['s1_1361_0'].values[i]
                df_merge['s1_insr_pY1361_mixed'].values[i]=df_merge['s1_insr_1361_0'].values[i]
        print(df_merge [(df_merge ['CodeC']==1457) & (df_merge ['CodeB']==210) & (df_merge ['CodeA']==104)])
        df_merge['total_pY1355']=0
        df_merge['total_pY1361']=0
        df_merge['total_insr']=0
        df_merge['s1_pY1355']=0
        df_merge['s1_pY1361']=0
        df_merge['s1_insr']=0
        for col in df_merge.columns:
            if "pY1355.erh_S1" in col:
                df_merge['s1_pY1355'] = df_merge['s1_pY1355']+df_merge[str(col)]
            elif "pY1355.erh" in col:
                df_merge['total_pY1355'] = df_merge['total_pY1355']+df_merge[str(col)]
            elif "pY1361.erh_S1" in col:
                df_merge['s1_pY1361'] = df_merge['s1_pY1361']+df_merge[str(col)]
            elif "pY1361.erh" in col:
                df_merge['total_pY1361'] = df_merge['total_pY1361']+df_merge[str(col)]
            elif "INSR.erh_S1" in col:
                df_merge['s1_insr'] = df_merge['s1_insr']+df_merge[str(col)]
            elif "INSR.erh" in col:
                df_merge['total_insr'] = df_merge['total_insr']+df_merge[str(col)]
        print(df_merge [(df_merge ['CodeC']==1457) & (df_merge ['CodeB']==210) & (df_merge ['CodeA']==104)])
        #df_erh_insr = df_merge[['CodeA', 'CodeB', 'CodeC','total_pY1355','total_pY1361','total_insr','total_pY1355_mixed','total_pY1361_mixed','total_insr_pY1355_mixed','total_insr_pY1361_mixed']]
        #df_erh_insr.fillna(0,inplace=True)
        #df_erh_insr=df_merge
        
        cols=['richness_1355_0', 'richness_1355_1', 'richness_1361_0', 'richness_1361_1', 'richness_insr_1355_0', 'richness_insr_1355_1', 'richness_insr_1361_0', 'richness_insr_1361_1', 's1_1355_0', 's1_1355_1', 's1_1361_0', 's1_1361_1', 's1_insr_1355_0', 's1_insr_1355_1', 's1_insr_1361_0', 's1_insr_1361_1']
        for col in df_merge.columns:
            if '.erh' in col:
                cols.append(col)
                cols.append('class')
        print('columns to be dropped are: ',cols)
        df_erh_insr =df_merge.drop(cols,axis=1)
        print(df_erh_insr.columns)

        print(df_erh_insr [(df_erh_insr ['CodeC']==192) & (df_erh_insr ['CodeB']==207) & (df_erh_insr ['CodeA']==134)])    
        print(df_erh_insr [(df_erh_insr ['CodeC']==192) & (df_erh_insr ['CodeB']==247) & (df_erh_insr ['CodeA']==169)])
        print(df_erh_insr [(df_erh_insr ['CodeC']==192) & (df_erh_insr ['CodeB']==245) & (df_erh_insr ['CodeA']==191)])
        print(df_erh_insr [(df_erh_insr ['CodeC']==192) & (df_erh_insr ['CodeB']==245) & (df_erh_insr ['CodeA']==235)])
        print(df_erh_insr [(df_erh_insr ['CodeC']==192) & (df_erh_insr ['CodeB']==207) & (df_erh_insr ['CodeA']==134)])
        print(df_erh_insr [(df_erh_insr ['CodeC']==192) & (df_erh_insr ['CodeB']==152) & (df_erh_insr ['CodeA']==173)])
        print(df_erh_insr [(df_erh_insr ['CodeC']==192) & (df_erh_insr ['CodeB']==247) & (df_erh_insr ['CodeA']==174)])
        print(df_erh_insr [(df_erh_insr ['CodeC']==192) & (df_erh_insr ['CodeB']==137) & (df_erh_insr ['CodeA']==233)])
        print(df_erh_insr [(df_erh_insr ['CodeC']==1) & (df_erh_insr ['CodeB']==17) & (df_erh_insr ['CodeA']==71)])
        print(df_erh_insr [(df_erh_insr ['CodeC']==192) & (df_erh_insr ['CodeB']==194) & (df_erh_insr ['CodeA']==202)])
        print(df_erh_insr [(df_erh_insr ['CodeC']==192) & (df_erh_insr ['CodeB']==194) & (df_erh_insr ['CodeA']==52)])
        print(df_erh_insr [(df_erh_insr ['CodeC']==192) & (df_erh_insr ['CodeB']==234) & (df_erh_insr ['CodeA']==173)])
        print(df_erh_insr [(df_erh_insr ['CodeC']==192) & (df_erh_insr ['CodeB']==245) & (df_erh_insr ['CodeA']==156)])
        return df_erh_insr



    def classification(df_filtered):
        x = np.array(df_filtered[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']])
        y_predict,model_name,score_of_determination = ClusteringAnalysis.oneclassSVM(x)
        #y_predict,model_name  = ClusteringAnalysis.kmeans(df_filtered)
        #y_predict,model_name = ClusteringAnalysis.oneclassSVM(df_filtered)
        #y_predict,model_name = ClusteringAnalysis.oneclassSVM_simple(df)
        #y_predict,model_name  = ClusteringAnalysis.spectral(df_filtered)

        #y_predict,model_name  = ClusteringAnalysis.birch(df_filtered)
        #y_predict,model_name  = ClusteringAnalysis.agglomerativeClustering(df_filtered)
        #y_predict,model_name  = ClusteringAnalysis.opticsClustering(df_filtered)
        df_result=pd.DataFrame()
        df_result['CodeA']=df_filtered['CodeA']
        df_result['CodeB']=df_filtered['CodeB']
        df_result['CodeC']=df_filtered['CodeC']
        df_result['Score_OSVM'] = score_of_determination
        df_result['class'] = y_predict
        df_result = df_result.sort_values(by =['Score_OSVM'],ascending=True)
        df_result['SCORE_RANK_OSVM']=df_result['Score_OSVM'].rank(ascending=True)
        #df_result=  df_result.sort_values(by =['Score','performance_ind_0_total','performance_ind_1_total'],ascending=False)
        #df_result['SCORE_RANK'] = df_result[['Score','performance_ind_0_total','performance_ind_1_total']].apply(tuple,axis=1).rank(method='dense',ascending=False)
        print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==247) & (df_result ['CodeA']==169)])
        print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==245) & (df_result ['CodeA']==191)])
        print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==245) & (df_result ['CodeA']==235)])
        print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==207) & (df_result ['CodeA']==134)])
        print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==152) & (df_result ['CodeA']==173)])
        print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==247) & (df_result ['CodeA']==174)])
        print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==137) & (df_result ['CodeA']==233)])
        print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==194) & (df_result ['CodeA']==202)])
        print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==194) & (df_result ['CodeA']==52)])
        print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==234) & (df_result ['CodeA']==173)])
        print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==245) & (df_result ['CodeA']==156)])
        print(df_result [(df_result ['CodeC']==1) & (df_result ['CodeB']==17) & (df_result ['CodeA']==71)])
        print(df_result [(df_result ['CodeC']==260) & (df_result ['CodeB']==26) & (df_result ['CodeA']==22)])
        result_name='output_'+model_name+'.csv'
        df_result.to_csv(output_dir+result_name)
        return y_predict,model_name,x,df_result 

    def ipcaAnalysis(df_erh_insr,df_result):
        gc.collect()
        print(df_erh_insr.columns)

        df_erh_insr=df_result.merge(df_erh_insr,how='left',on=['CodeA','CodeB','CodeC'])
        df_erh_insr_ = df_erh_insr[['total_pY1355','total_pY1361','total_insr','total_pY1355_mixed','total_pY1361_mixed','total_insr_pY1355_mixed','total_insr_pY1361_mixed','s1_pY1355','s1_pY1361','s1_insr','s1_pY1355_mixed','s1_pY1361_mixed','s1_insr_pY1355_mixed','s1_insr_pY1361_mixed']]
        df_erh_insr_normalized =  pd.DataFrame(scaler.fit_transform(df_erh_insr_.values),columns=df_erh_insr_.columns+'_normalized',index=df_erh_insr_.index)

        df_erh_insr_ipca = pd.concat((df_erh_insr,df_erh_insr_normalized),axis=1)
        x_normalized_1355 = np.array(df_erh_insr_ipca[['total_insr_normalized','total_pY1355_normalized']])    
 
        #x_normalized_1355 = np.array(df_erh_insr_ipca[['total_insr','total_pY1355']])    
        #x_normalized_1361 = np.array(df_erh_insr_ipca[['total_insr','total_pY1361']])   
        x_ipca_1355 = ClusteringAnalysis.ipca(x_normalized_1355,2 )
        print(x_ipca_1355)
        df_erh_insr_ipca['total_insr_1355_ipca'] = x_ipca_1355[:,0] 
        df_erh_insr_ipca['total_pY1355_ipca'] = x_ipca_1355[:,1] 
        x_normalized_1361 = np.array(df_erh_insr_ipca[['total_insr_normalized','total_pY1361_normalized']])  
        x_ipca_1361 = ClusteringAnalysis.ipca(x_normalized_1361,2 )
        df_erh_insr_ipca['total_insr_1361_ipca'] = x_ipca_1361[:,0] 
        df_erh_insr_ipca['total_pY1361_ipca'] = x_ipca_1361[:,1] 
        x_normalized_1355_mixed = np.array(df_erh_insr_ipca[['total_insr_pY1355_mixed_normalized','total_pY1355_mixed_normalized']]) 

        #x_normalized_1355_mixed = np.array(df[['total_insr','total_pY1355']])    
        #x_normalized_1361_mixed = np.array(df[['total_insr','total_pY1361']])   
        x_ipca_1355_mixed= ClusteringAnalysis.ipca(x_normalized_1355_mixed,2 )
        df_erh_insr_ipca['total_insr_pY1355_mixed_ipca'] = x_ipca_1355_mixed[:,0] 
        df_erh_insr_ipca['total_pY1355_mixed_ipca'] = x_ipca_1355_mixed[:,1] 
        x_normalized_1361_mixed = np.array(df_erh_insr_ipca[['total_insr_pY1361_mixed_normalized','total_pY1361_mixed_normalized']])  
        x_ipca_1361_mixed = ClusteringAnalysis.ipca(x_normalized_1361_mixed,2 )
        df_erh_insr_ipca['total_insr_pY1361_mixed_ipca'] = x_ipca_1361_mixed[:,0] 
        df_erh_insr_ipca['total_pY1361_mixed_ipca'] = x_ipca_1361_mixed[:,1]

        x_erh_1355_insr_normalized = np.array(df_erh_insr_ipca[['total_pY1355_normalized','s1_pY1355_normalized','total_insr_normalized','s1_insr_normalized']])    
 
        #x_normalized_1355 = np.array(df_erh_insr_ipca[['total_insr','total_pY1355']])    
        #x_normalized_1361 = np.array(df_erh_insr_ipca[['total_insr','total_pY1361']])   
        x_erh_1355_ipca = ClusteringAnalysis.ipca(x_erh_1355_insr_normalized ,3 )
        print(x_erh_1355_ipca.shape)
        df_erh_insr_ipca['pY1355_ipca_x'] = x_erh_1355_ipca[:,0] 
        df_erh_insr_ipca['pY1355_ipca_y'] =  x_erh_1355_ipca[:,1] 
        df_erh_insr_ipca['pY1355_ipca_z'] =  x_erh_1355_ipca[:,2] 

        x_erh_1361_insr_normalized = np.array(df_erh_insr_ipca[['total_pY1361_normalized','s1_pY1361_normalized','total_insr_normalized','s1_insr_normalized']])    
 
        #x_normalized_1355 = np.array(df_erh_insr_ipca[['total_insr','total_pY1355']])    
        #x_normalized_1361 = np.array(df_erh_insr_ipca[['total_insr','total_pY1361']])   
        x_erh_1361_ipca = ClusteringAnalysis.ipca(x_erh_1361_insr_normalized ,3 )
        print(x_erh_1361_ipca.shape)
        df_erh_insr_ipca['pY1361_ipca_x'] = x_erh_1361_ipca[:,0] 
        df_erh_insr_ipca['pY1361_ipca_y'] =  x_erh_1361_ipca[:,1] 
        df_erh_insr_ipca['pY1361_ipca_z'] =  x_erh_1361_ipca[:,2] 

        x_erh_insr_1355_normalized_mixed = np.array(df_erh_insr_ipca[['total_pY1355_mixed_normalized','s1_pY1355_mixed_normalized','total_insr_pY1355_mixed_normalized','s1_insr_pY1355_mixed_normalized']]) 

        #x_normalized_1355_mixed = np.array(df[['total_insr','total_pY1355']])    
        #x_normalized_1361_mixed = np.array(df[['total_insr','total_pY1361']])   
        x_erh_1355_ipca_mixed= ClusteringAnalysis.ipca(x_erh_insr_1355_normalized_mixed,3 )
        df_erh_insr_ipca['pY1355_mixed_ipca_x'] = x_erh_1355_ipca_mixed[:,0] 
        df_erh_insr_ipca['pY1355_mixed_ipca_y'] = x_erh_1355_ipca_mixed[:,1] 
        df_erh_insr_ipca['pY1355_mixed_ipca_z'] = x_erh_1355_ipca_mixed[:,2] 
        x_erh_insr_1361_normalized_mixed = np.array(df_erh_insr_ipca[['total_pY1361_mixed_normalized','s1_pY1361_mixed_normalized','total_insr_pY1361_mixed_normalized','s1_insr_pY1361_mixed_normalized']]) 

        #x_normalized_1355_mixed = np.array(df[['total_insr','total_pY1355']])    
        #x_normalized_1361_mixed = np.array(df[['total_insr','total_pY1361']])   
        x_erh_1361_ipca_mixed= ClusteringAnalysis.ipca(x_erh_insr_1361_normalized_mixed,3 )
        df_erh_insr_ipca['pY1361_mixed_ipca_x'] = x_erh_1361_ipca_mixed[:,0] 
        df_erh_insr_ipca['pY1361_mixed_ipca_y'] = x_erh_1361_ipca_mixed[:,1] 
        df_erh_insr_ipca['pY1361_mixed_ipca_z'] = x_erh_1361_ipca_mixed[:,2] 

        return df_erh_insr_ipca
    def similarity(df_erh_insr_ipca,model_name):
        gc.collect()
        model_names = ['kmeans','AgglmerativeClustering']
        #print(df_result.columns)

        #print(df_erh_insr_ipca.columns)
        #df_ = df_erh_insr_ipca.drop('class',axis=1)
        #df= df_result.merge(df_erh_insr_ipca,how='left',on=['CodeA','CodeB','CodeC'])
        df = df_erh_insr_ipca
        print(df.columns)


        y_min_1355 = min(df['total_pY1355_ipca'] )
        y_min_1361 = min(df['total_pY1361_ipca'] )
        x_min_1355 = min(df['total_insr_1355_ipca'])
        x_min_1361= min(df['total_insr_1361_ipca'])
        x_argmin_1355 = df['total_insr_1355_ipca']  [np.argmin(df['total_pY1355_ipca'])]
        x_argmin_1361 = df['total_insr_1361_ipca']  [np.argmin(df['total_pY1361_ipca'])]   
        y_argmin_1355 = df['total_pY1355_ipca']  [np.argmin(df['total_insr_1355_ipca'])]
        y_argmin_1361 = df['total_pY1361_ipca']  [np.argmin(df['total_insr_1361_ipca'])]   
        x_min = min(df['total_insr'])
        print(x_min_1355,y_min_1355)
        print(x_min_1361,y_min_1361 )
        erh_array_1355 = np.array(df[['total_insr_1355_ipca','total_pY1355_ipca']])
        erh_array_1361 = np.array(df[['total_insr_1361_ipca','total_pY1361_ipca']])
        df_xmin_pY1355= df[df['total_insr_1355_ipca']<x_argmin_1355]
        df_xymin_pY1355 = df_xmin_pY1355[df_xmin_pY1355['total_pY1355_ipca']<y_argmin_1355]
        if len(df_xymin_pY1355)>0:
            ct_ind_pY1355= 1
        else:
            ct_ind_pY1355 = 0
        df_xmin_pY1361= df[df['total_insr_1361_ipca']<x_argmin_1361]
        df_xymin_pY1361 = df_xmin_pY1361[df_xmin_pY1361['total_pY1361_ipca']<y_argmin_1361]
        if len(df_xymin_pY1361)>0:
            ct_ind_pY1361= 1
        else:
            ct_ind_pY1361 = 0
        x_min = min(df['total_insr'])
        #df['Similarity_point_pY1355'] = utils.distanceCP(erh_array_1355,c_1355)
        #df['Similarity_point_pY1361'] = utils.distanceCP(erh_array_1361,c_1361)
        if ct_ind_pY1355 == 1:
            x_min_1355_ipca = x_argmin_1355 
            df['Similarity_centralLine_pY1355'] = ( df['total_insr_1355_ipca']-x_min_1355_ipca)
            c_1355 = [x_min_1355_ipca,y_min_1355]
            df['Similarity_point_pY1355'] = utils.distanceCP(erh_array_1355,c_1355)
        else:
            y_min_1355_ipca = df['total_pY1355_ipca']  [np.argmin(df['total_insr_1355_ipca'])]
            df['Similarity_centralLine_pY1355'] = ( df['total_pY1355_ipca']-y_min_1355_ipca)
            c_1355 = [x_min_1355,y_min_1355_ipca]
            df['Similarity_point_pY1355'] = utils.distanceCP(erh_array_1355,c_1355)
        if ct_ind_pY1361 == 1:
            x_min_1361_ipca = x_argmin_1361
            df['Similarity_centralLine_pY1361'] = ( df['total_insr_1361_ipca']-x_min_1361_ipca)
            c_1361 = [x_min_1361_ipca,y_min_1361]   
            print(c_1361 )
            df['Similarity_point_pY1361'] = utils.distanceCP(erh_array_1361,c_1361)
        else:
            y_min_1361_ipca = df['total_pY1361_ipca']  [np.argmin(df['total_insr_1361_ipca'])]
            df['Similarity_centralLine_pY1361'] = ( df['total_pY1361_ipca']-y_min_1361_ipca)
            c_1361 = [x_min_1361,y_min_1361_ipca]   
            df['Similarity_point_pY1361'] = utils.distanceCP(erh_array_1361,c_1361)
        df['Similarity_pY1355']=df['Similarity_point_pY1355'] + abs(df['Similarity_centralLine_pY1355'])
        df['Similarity_pY1361']=df['Similarity_point_pY1361'] + abs(df['Similarity_centralLine_pY1361'])

        y_min_1355_mixed = min(df['total_pY1355_mixed_ipca'])
        y_min_1361_mixed= min(df['total_pY1361_mixed_ipca'])
        x_min_1355_mixed = min(df['total_insr_pY1355_mixed_ipca'])
        x_min_1361_mixed= min(df['total_insr_pY1361_mixed_ipca'])

        x_argmin_1355_mixed =  df['total_insr_pY1355_mixed_ipca']  [np.argmin(df['total_pY1355_mixed_ipca'])]
        x_argmin_1361_mixed= df['total_insr_pY1361_mixed_ipca']  [np.argmin(df['total_pY1361_mixed_ipca'])]   
        y_argmin_1355_mixed = df['total_pY1355_mixed_ipca']  [np.argmin(df['total_insr_1355_ipca'])]
        y_argmin_1361_mixed = df['total_pY1361_mixed_ipca']  [np.argmin(df['total_insr_1361_ipca'])]   
        erh_array_1355_mixed = np.array(df[['total_insr_pY1355_mixed_ipca','total_insr_pY1355_mixed_ipca']])
        erh_array_1361_mixed = np.array(df[['total_insr_pY1361_mixed_ipca','total_insr_pY1361_mixed_ipca']])
        print(x_min_1355_mixed,x_min_1361_mixed,y_min_1355_mixed,y_min_1361_mixed)
        print(x_min_1355_mixed,x_min_1361_mixed,x_argmin_1355_mixed,y_min_1361_mixed,x_argmin_1361_mixed)
        df_xmin_pY1355_mixed= df[df['total_insr_pY1355_mixed_ipca']<x_argmin_1355_mixed]
        df_xymin_pY1355_mixed = df_xmin_pY1355_mixed[df_xmin_pY1355_mixed['total_pY1355_mixed_ipca']<y_argmin_1355_mixed]
        if len(df_xymin_pY1355_mixed)>0:
            ct_ind_pY1355_mixed= 1
        else:
            ct_ind_pY1355_mixed = 0
        df_xmin_pY1361_mixed= df[df['total_insr_pY1361_mixed_ipca']<x_argmin_1361_mixed]
        df_xymin_pY1361_mixed = df_xmin_pY1361_mixed[df_xmin_pY1361_mixed['total_pY1361_mixed_ipca']<y_argmin_1361_mixed]
        if len(df_xymin_pY1361_mixed)>0:
            ct_ind_pY1361_mixed= 1
        else:
            ct_ind_pY1361_mixed = 0


        
        if ct_ind_pY1355_mixed == 1:

            df['Similarity_antiinsr_1355'] = 1/(df['total_insr_pY1355_mixed_ipca']-x_min_1355_mixed)
            y_min_1355_mixed_ipca = y_min_1355_mixed 
            x_min_1355_mixed_ipca = x_argmin_1355_mixed 
            df['Similarity_centralLine_pY1355_mixed_ipca'] = ( df['total_insr_pY1355_mixed_ipca']-x_min_1355_mixed_ipca)
            c_1355_mixed = [x_min_1355_mixed_ipca,y_min_1355_mixed_ipca]
            df['Similarity_point_pY1355_mixed'] = utils.distanceCP(erh_array_1355_mixed,c_1355_mixed)
            print('central line is along fixed x ',  x_min_1355_mixed_ipca )
        else:

            df['Similarity_antiinsr_1355'] = 1/(df['total_pY1355_mixed_ipca']-y_min_1355_mixed)
            x_min_1355_mixed_ipca = x_min_1355_mixed
            y_min_1355_mixed_ipca = df['total_pY1355_mixed_ipca']  [np.argmin(df['total_insr_pY1355_mixed_ipca'])]
            
            df['Similarity_centralLine_pY1355_mixed_ipca'] = ( df['total_pY1355_mixed_ipca']-y_min_1355_mixed_ipca)
            c_1355_mixed = [x_min_1355_mixed_ipca,y_min_1355_mixed_ipca]
            df['Similarity_point_pY1355_mixed'] = utils.distanceCP(erh_array_1355_mixed,c_1355_mixed)
            print('central line is along fixed y ', y_min_1355_mixed_ipca )
        if ct_ind_pY1361_mixed == 1:

            df['Similarity_antiinsr_1361'] = 1/(df['total_insr_pY1361_mixed_ipca']-x_min_1361_mixed)
            y_min_1361_mixed_ipca = y_min_1361_mixed 
            x_min_1361_mixed_ipca = x_argmin_1361_mixed
            df['Similarity_centralLine_pY1361_mixed_ipca'] = ( df['total_insr_pY1361_mixed_ipca']-x_min_1361_mixed_ipca)
            c_1361_mixed = [x_min_1361_mixed_ipca,y_min_1361_mixed_ipca]   
            print('central line is along fixed x ',  x_min_1361_mixed_ipca )
            print(c_1361_mixed )
            df['Similarity_point_pY1361_mixed'] = utils.distanceCP(erh_array_1361_mixed,c_1361_mixed)
        else:

            df['Similarity_antiinsr_1361'] = 1/(df['total_pY1361_mixed_ipca']-y_min_1355_mixed)
            x_min_1361_mixed_ipca =  x_min_1361_mixed
            y_min_1361_mixed_ipca = df['total_pY1361_mixed_ipca']  [np.argmin(df['total_insr_pY1361_mixed_ipca'])]
            df['Similarity_centralLine_pY1361_mixed_ipca'] = ( df['total_pY1361_mixed_ipca']-y_min_1361_mixed_ipca)
            c_1361_mixed = [x_min_1361_mixed_ipca,y_min_1361_mixed_ipca]   
            df['Similarity_point_pY1361_mixed'] = utils.distanceCP(erh_array_1361_mixed,c_1361_mixed)
            print('central line is along fixed y ', y_min_1361_mixed_ipca )
        df_0 = df[df['class']==0]
        df_1 = df[df['class']==1]
        x_0= df_0[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']]
        x_0=np.array(x_0)
        x_1= df_1[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']]
        x_1=np.array(x_1)
        if model_name=='OneClassSVM':
            df_0['Similarity']=df_0['Score_OSVM']
            df_1['Similarity']=df_1['Score_OSVM']        
        else:
            df_0['Similarity']=utils.boundary(x_0,x_1)
            df_1['Similarity']=utils.boundary(x_1,x_0)
        df_similarity=pd.concat((df_0,df_1),axis=0)
        print(model_name, df_0['Similarity'])

        return df_similarity
  

class ClusteringAnalysis():
    def logProbability(x):
        #print(model)
        #model_orig = OneClassSVM(kernel='rbf',gamma='auto',nu=0.8,coef0=0.1,tol=1e-5)
        #x = np.reshape(x,(len(x),1))
        #model = model_orig.fit(x)
        score = model.score_samples(x)
        score[score==0]=1
        #print(score[score==0])
        f = np.log(score)
        #print(f[f==np.inf])
        #print(f[f==-np.inf])

        return f
    def opticsClustering(df):
        gc.collect()
        #x = np.array(df[['Richness_SUM','Richness_COUNT','S1_SUM','S1_COUNT']])
        x = np.array(df[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']])
        df_result = df
        ordering,core_distances,reachability,predecessor=compute_optics_graph(x,min_samples=2,max_eps=np.inf,metric='euclidean',p=2,metric_params=None,algorithm='auto',leaf_size=2,n_jobs=None)
        labels,clusters=cluster_optics_xi(reachability=reachability,predecessor=predecessor,ordering=ordering,min_samples=2,xi=0.00001)
        #model = OPTICS(metric='euclidean')
        #x = np.reshape(x,(len(x),1))
        #model = model.fit(x)
        #y_predict = model.labels_
        y_predict = labels
        df_result['class'] = y_predict
        print(df [(df ['CodeC']==192) & (df ['CodeB']==247) & (df ['CodeA']==169)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==191)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==235)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==207) & (df ['CodeA']==134)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==152) & (df ['CodeA']==173)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==247) & (df ['CodeA']==174)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==137) & (df ['CodeA']==233)])
        print(df [(df ['CodeC']==1) & (df ['CodeB']==17) & (df ['CodeA']==71)])
        df_result.to_csv(output_dir+'labels_opticsClustering.csv',chunksize=1000)
        #score_of_determination = model.score(x,y)

        #print('Score of model is: ',score_of_determination)
        model_name='optics'
        result_name='output_'+model_name+'.csv'
        df_result.to_csv(output_dir+result_name)
        return y_predict,model_name,x,df_result
    def kmeans(df):
        gc.collect()
        #x = np.array(df[['Richness_SUM','Richness_COUNT','Richness_STDEV']])
        #x = np.array(df[['S1_ind','Richness_ind','S1_Richness_balance','S1_Richness_efficiency']])
        #x = np.array(df[['Richness_SUM','Richness_COUNT','S1_SUM','S1_COUNT']])
        x = np.array(df[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']])
        df_result = df
        model = KMeans(n_clusters=2,init='k-means++',algorithm='lloyd',tol=5e-4)
        #x = np.reshape(x,(len(x),1))
        model = model.fit(x)
        y_predict = model.predict(x)
        df_result['class'] = y_predict
        print(df [(df ['CodeC']==192) & (df ['CodeB']==247) & (df ['CodeA']==169)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==191)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==235)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==207) & (df ['CodeA']==134)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==152) & (df ['CodeA']==173)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==247) & (df ['CodeA']==174)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==137) & (df ['CodeA']==233)])
        print(df [(df ['CodeC']==1) & (df ['CodeB']==17) & (df ['CodeA']==71)])
        df_result.to_csv(output_dir+'labels_kmeans.csv',chunksize=1000)
        #y_predict = model.predict(x)
        score_of_determination = model.score(x)

        print('Score of model is: ',score_of_determination)
        model_name='kmeans'
        result_name='output_'+model_name+'.csv'
        df_result.to_csv(output_dir+result_name)
        return y_predict,model_name,x,df_result
    
    def oneclassSVM(x):
        global model
        #df['Richness_STDEV']=1/df['Richness_STDEV']
        #df['S1_STDEV']=1/df['S1_STDEV']
        #x = np.array(df[['S1_ind','Richness_ind','S1_Richness_balance','S1_Richness_efficiency']])
        #x = np.array(df[['Richness_SUM','Richness_COUNT','S1_SUM','S1_COUNT']])


        #model = SVR(kernel='linear',gamma='scale',tol=0.001,epsilon=0.01)
        #model = SVC(kernel='linear',gamma='scale',tol=0.0005,epsilon=0.0001)
        model = OneClassSVM(kernel='rbf',gamma='auto',nu=0.8,coef0=0.1,tol=1e-5)
        #x = np.reshape(x,(len(x),1))
        model = model.fit(x)
        y_predict = model.predict(x)
        score_of_determination = model.score_samples(x)
        #x_shap_orig = pd.DataFrame(x,columns=['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT'])
        x_shap_orig = pd.DataFrame(x,columns=['Enrichment_SUM','Enrichment_STDEV','Enrichment_COUNT','S1_SUM','S1_STDEV','S1_COUNT'])
        x_shap = shap.sample(x_shap_orig,1000)
        x_shap.to_csv('oneClass_x.csv')
        
        explainer_kernel = shap.KernelExplainer(ClusteringAnalysis.logProbability,x_shap)
        #explainer_kernel = shap.KernelExplainer(model.score_samples,x_shap)
        shap_values_kernel=explainer_kernel(x_shap)
        expected_value_kernel = explainer_kernel.expected_value
        print(expected_value_kernel)
        print(shap_values_kernel)
        print(shap_values_kernel[0])
        plt.figure()
        #plt.title('Clustering') 
        shap.plots.waterfall(shap_values_kernel[0],show=False)        
        fig22=plt.gcf()
        fig22.legend()
        png_name='OCSVM_kernel_waterfall_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name)
        plt.close() 




        plt.figure()
        shap.summary_plot(shap_values_kernel,x_shap,show=False)        
        fig24=plt.gcf()
        fig24.legend()
        png_name='OCSVM_kernel_summaryplot_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name)
        plt.close() 
        
        plt.figure()

        #plt.title('Clustering') 

        shap.plots.scatter(shap_values_kernel[:,'Enrichment_SUM'],color=shap_values_kernel[:,'Enrichment_COUNT'],show=False)        
        fig25=plt.gcf()
        fig25.legend()
        png_name='OCSVM_kernel_scatter_Enrichmentsum_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name)
        plt.close() 

        shap.plots.scatter(shap_values_kernel[:,'S1_SUM'],color=shap_values_kernel,show=False)        
        fig26=plt.gcf()
        fig26.legend()
        png_name='OCSVM_kernel_scatter_s1sum_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name)
        plt.close() 

        shap.plots.scatter(shap_values_kernel[:,'S1_STDEV'],color=shap_values_kernel,show=False)        
        fig27=plt.gcf()
        fig27.legend()
        png_name='OCSVM_kernel_scatter_s1stdev_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name)
        plt.close() 

        shap.plots.scatter(shap_values_kernel[:,'Enrichment_STDEV'],color=shap_values_kernel,show=False)        
        fig28=plt.gcf()
        fig28.legend()
        png_name='OCSVM_kernel_scatter_Enrichmentstdev_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name)
        plt.close() 

        shap.plots.scatter(shap_values_kernel[:,'Enrichment_COUNT'],color=shap_values_kernel,show=False)        
        fig29=plt.gcf()
        fig29.legend()
        png_name='OCSVM_kernel_scatter_Enrichmentcount_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name)
        plt.close() 

        shap.plots.scatter(shap_values_kernel[:,'S1_COUNT'],color=shap_values_kernel,show=False)        
        fig30=plt.gcf()
        fig30.legend()
        png_name='OCSVM_kernel_scatter_s1count_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name)
        plt.close() 


        plt.figure()
        png_name='OCSVM_kernel_forceplot_'+str(currentTime)+'.png'
        #shap.plots.force(shap_values_kernel[0],matplotlib=True,show=False,plot_cmap='DrDb')   
        
        #shap.force_plot(np.round(expected_value_kernel,4),np.round(shap_values_kernel.values[0,:],4),np.round(x_shap.iloc[0,:],4),matplotlib=True,show=False,plot_cmap='DrDb').savefig(graph_dir+png_name)   
        shap.force_plot(np.round(expected_value_kernel,4),np.round(shap_values_kernel.values[0,:],4),np.round(x_shap.iloc[0,:],4),matplotlib=True,show=False,plot_cmap='DrDb').savefig(graph_dir+png_name)  
        #plt.show()
        plt.close()
        
        plt.figure()
        #plt.title('Clustering') 
        shap.plots.heatmap(shap_values_kernel,show=False)        
        fig23=plt.gcf()
        fig23.legend()
        png_name='OCSVM_kernel_heatmap_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name)
        plt.close() 

        model_orig = OneClassSVM(kernel='rbf',gamma='auto',nu=0.8,coef0=0.1,tol=1e-5)
        model_shap = model_orig.fit(x_shap)
        explainer = shap.Explainer(model_shap.predict,x_shap)
        shap_values=explainer(x_shap)
        print(shap_values)
        print(shap_values[0])
        plt.figure()
        #plt.title('Clustering') 
        shap.plots.waterfall(shap_values[0],show=False)        
        fig18=plt.gcf()
        fig18.legend()
        png_name='OCSVM_waterfall_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name)
        plt.close() 

        plt.figure()
        shap.plots.force(explainer.expected_value,shap_values,matplotlib=True,show=False,plot_cmap='DrDb')
        fig36=plt.gcf()
        fig36.legend()
        png_name='OCSVM_kernel_forceplot_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name)
        plt.close() 

        plt.figure()
        #plt.title('Clustering') 
        shap.plots.heatmap(shap_values,show=False)        
        fig20=plt.gcf()
        fig20.legend()
        png_name='OCSVM_heatmap_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name)
        plt.close() 

        plt.figure()
        shap.summary_plot(shap_values,x_shap,show=False)        
        fig21=plt.gcf()
        fig21.legend()
        png_name='OCSVM_summaryplot_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name)
        plt.close() 

        plt.figure()

        #plt.title('Clustering') 

        shap.plots.scatter(shap_values[:,'Enrichment_SUM'],color=shap_values[:,'Enrichment_COUNT'],show=False)       
        fig19=plt.gcf()
        fig19.legend()
        png_name='OCSVM_scatter_Enrichmentsum_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name)
        plt.close() 

        shap.plots.scatter(shap_values[:,'S1_SUM'],color=shap_values,show=False)        
        fig31=plt.gcf()
        fig31.legend()
        png_name='OCSVM_scatter_s1sum_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name)
        plt.close() 

        shap.plots.scatter(shap_values[:,'S1_STDEV'],color=shap_values,show=False)        
        fig32=plt.gcf()
        fig32.legend()
        png_name='OCSVM_scatter_s1stdev_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name)
        plt.close() 

        shap.plots.scatter(shap_values[:,'Enrichment_STDEV'],color=shap_values,show=False)        
        fig33=plt.gcf()
        fig33.legend()
        png_name='OCSVM_scatter_Enrichmentstdev_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name)
        plt.close() 

        shap.plots.scatter(shap_values[:,'Enrichment_COUNT'],color=shap_values,show=False)        
        fig34=plt.gcf()
        fig34.legend()
        png_name='OCSVM_scatter_Enrichmentcount_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name)
        plt.close() 

        shap.plots.scatter(shap_values[:,'S1_COUNT'],color=shap_values,show=False)        
        fig35=plt.gcf()
        fig35.legend()
        png_name='OCSVM_scatter_s1count_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name)
        plt.close() 

        y_predict[y_predict==1]=0
        y_predict[y_predict==-1]=1

        print('Score of model is: ',score_of_determination)
        model_name = 'OneClassSVM'


        return y_predict,model_name,score_of_determination 
    

    def agglomerativeClustering(df):
        gc.collect()
        x = np.array(df[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']])
        #x = np.array(df[['S1_ind','Richness_ind','S1_Richness_balance','S1_Richness_efficiency']])
        df_result = df
        #model = SVR(kernel='linear',gamma='scale',tol=0.001,epsilon=0.01)
        #model = SVC(kernel='linear',gamma='scale',tol=0.0005,epsilon=0.0001)
        #model = OneClassSVM(kernel='rbf',gamma='auto',tol=0.001)
        model = AgglomerativeClustering(n_clusters=2,metric='euclidean',linkage='ward',compute_distances=True)
        #x = np.reshape(x,(len(x),1))
        model = model.fit(x)
        print('distance is: ',model.distances_)
        print(np.mean(model.distances_),max(model.distances_),model.n_clusters_)
        y_predict = model.labels_
        df_result['class'] = y_predict
        print(df [(df ['CodeC']==192) & (df ['CodeB']==247) & (df ['CodeA']==169)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==191)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==235)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==207) & (df ['CodeA']==134)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==152) & (df ['CodeA']==173)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==247) & (df ['CodeA']==174)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==137) & (df ['CodeA']==233)])
        print(df [(df ['CodeC']==1) & (df ['CodeB']==17) & (df ['CodeA']==71)])
        df_result.to_csv(output_dir+'labels_agglomerative.csv',chunksize=1000)
        model_name='AgglmerativeClustering'
        #y_predict = model.predict(x)
        #score_of_determination = model.score(x)
        #score_of_determination = model.score(x,y)
        result_name='output_'+model_name+'.csv'
        df_result.to_csv(output_dir+result_name)
        #print('Score of model is: ',score_of_determination)
        return y_predict,model_name,x,df_result
    def birch(df):
        gc.collect()
        #x = np.array(df[['Richness_SUM','Richness_COUNT','S1_SUM','S1_COUNT']])
        x = np.array(df[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']])
        #x = np.array(df[['S1_ind','Richness_ind','S1_Richness_balance','S1_Richness_efficiency']]) 
        df_result = df
        #model = SVR(kernel='linear',gamma='scale',tol=0.001,epsilon=0.01)
        #model = SVC(kernel='linear',gamma='scale',tol=0.0005,epsilon=0.0001)
        #model = OneClassSVM(kernel='rbf',gamma='auto',tol=0.001)
        model = Birch(threshold=0.7819798519436253,branching_factor=40,n_clusters=2)
        #x = np.reshape(x,(len(x),1))
        model = model.fit(x)
        y_predict = model.predict(x)

        df_result['class'] = y_predict
        print(df [(df ['CodeC']==192) & (df ['CodeB']==247) & (df ['CodeA']==169)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==191)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==245) & (df ['CodeA']==235)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==207) & (df ['CodeA']==134)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==152) & (df ['CodeA']==173)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==247) & (df ['CodeA']==174)])
        print(df [(df ['CodeC']==192) & (df ['CodeB']==137) & (df ['CodeA']==233)])
        print(df [(df ['CodeC']==1) & (df ['CodeB']==17) & (df ['CodeA']==71)])

        model_name = 'birch'
        #y_predict = model.predict(x)
        #score_of_determination = model.score(x)
        #score_of_determination = model.score(x,y)
        result_name='output_'+model_name+'.csv'
        df_result.to_csv(output_dir+result_name)
        #print('Score of model is: ',score_of_determination)
        return y_predict,model_name,x,df_result
    def spectral(df):
        gc.collect()
        #x = np.array(df[['Richness_SUM','Richness_COUNT','Richness_STDEV','S1_SUM','S1_COUNT','S1_STDEV']])
        x = np.array(df[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']])
        #df['S1_STDEV']=1/df['S1_STDEV']
        #df['Richness_STDEV']=1/df['Richness_STDEV']
        #x = np.array(df[['Richness_SUM','Richness_COUNT','S1_SUM','S1_COUNT']])        
        df_result = df
        model = SpectralClustering(n_clusters=2,assign_labels='kmeans',eigen_solver='arpack',random_state=0,affinity='nearest_neighbors')
        #x = np.reshape(x,(len(x),1))
        model = model.fit(x)
        y_predict = model.labels_
        df_result['class'] = y_predict
        print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==247) & (df_result ['CodeA']==169)])
        print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==245) & (df_result ['CodeA']==191)])
        print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==245) & (df_result ['CodeA']==235)])
        print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==207) & (df_result ['CodeA']==134)])
        print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==152) & (df_result ['CodeA']==173)])
        print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==247) & (df_result ['CodeA']==174)])
        print(df_result [(df_result ['CodeC']==192) & (df_result ['CodeB']==137) & (df_result ['CodeA']==233)])
        print(df_result [(df_result ['CodeC']==1) & (df_result ['CodeB']==17) & (df_result ['CodeA']==71)])
        model_name='spectral'

        result_name='output_'+model_name+'.csv'
        df_result.to_csv(output_dir+result_name)
        return y_predict,model_name,x,df_result

    def ipca(x,num):

        clf_ipca = IncrementalPCA(n_components=num,batch_size=200)
        clf_ipca  = clf_ipca.fit(x)
        x_ipca = clf_ipca.transform(x)

        return x_ipca
    def linearRegression(df_erh):
        scaler = StandardScaler()
        x = df_erh['total_insr']

        for col in df_erh.columns:
            if 'pY1355' in col:
                name = col
                y = df_erh[col]
            elif 'pY1361' in col:
                name = col
                y = df_erh[col]

        x = np.reshape(x,(x.shape[0],1))
        y = np.reshape(y,(y.shape[0],1)) 
        x = scaler.fit_transform(x)
        y = scaler.fit_transform(y)      
        print(x,y)
        model = LinearRegression()
        model.fit(x,y)
        y_predict = model.predict(x)
        lr_score = model.score(x,y)
        df_erh['total_insr_normalized']=x
        df_erh[name+'_normalized']=y
        df_erh['predict']=y_predict
        df_erh['lr_score'] = lr_score
        print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==247) & (df_erh ['CodeA']==169)])
        print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==245) & (df_erh ['CodeA']==191)])
        print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==245) & (df_erh ['CodeA']==235)])
        print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==207) & (df_erh ['CodeA']==134)])
        print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==152) & (df_erh ['CodeA']==173)])
        print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==247) & (df_erh ['CodeA']==174)])
        print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==137) & (df_erh ['CodeA']==233)])
        print(df_erh [(df_erh ['CodeC']==1) & (df_erh ['CodeB']==17) & (df_erh ['CodeA']==71)])
        print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==137) & (df_erh ['CodeA']==233)])
        print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==194) & (df_erh ['CodeA']==202)])
        print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==194) & (df_erh ['CodeA']==52)])
        print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==234) & (df_erh ['CodeA']==173)])        
        print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==245) & (df_erh ['CodeA']==156)])  
        
        return df_erh

class utils():
    def boundary(x,x_):
        similarity=[]
        for i in range (len(x)):
            similarity_=[]
            for j in range(len(x_)):
                #print(i,j)
                #dist_pair=cdist(np.reshape(x[i],(len(x[i]),1)),np.reshape(x_[j],(len(x[j]),1)),'euclidean')
                dist=euclidean(x[i],x_[j])
                similarity_point = 1/dist
                similarity_.append(similarity_point)
                #print(dist_)
                #print(len(similarity_))
            average_similarity=np.average(similarity_)
            similarity.append(average_similarity)
            #print(len(similarity))
        #print(len(similarity))
        return similarity
    def distanceCP(x,c):
        
        print(len(x))
        dist=[]
        for i in range (len(x)):
            try:
            
                dist_point=1/euclidean(x[i],c)
                dist.append(dist_point)
            except ZeroDivisionError:
                dist.append(1)
        return dist



class evaluation():
    def Visualize(x,df_similarity,y_predict):
        #scaler=MinMaxScaler()
        #scaler.fit(x)
        #x=scaler.transform(x)
        #x = np.array(df_result[['S1_ind','Richness_ind','S1_Richness_balance','S1_Richness_efficiency']])
        currentTime=str(datetime.datetime.now()).replace(':','_')
        x_ipca_1355 = np.array(df_similarity[['total_insr_pY1355_mixed_ipca','total_pY1355_mixed_ipca']])
        x_ipca_1361 = np.array(df_similarity[['total_insr_pY1361_mixed_ipca','total_pY1361_mixed_ipca']])
        x_erh_1355 = np.array(df_similarity[['total_insr_pY1355_mixed','total_pY1355_mixed']])    
        x_erh_1361 = np.array(df_similarity[['total_insr_pY1361_mixed','total_pY1361_mixed']])   
        x_antiinsr_pY1355_ipca = np.array(df_similarity[['pY1355_ipca_x','pY1355_ipca_y','pY1355_ipca_z']])   
        x_antiinsr_pY1361_ipca = np.array(df_similarity[['pY1361_ipca_x','pY1361_ipca_y','pY1355_ipca_z']])  
        x_antiinsr_pY1355_ipca_mixed = np.array(df_similarity[['pY1355_mixed_ipca_x','pY1355_mixed_ipca_y','pY1355_mixed_ipca_z']])   
        x_antiinsr_pY1361_ipca_mixed = np.array(df_similarity[['pY1361_mixed_ipca_x','pY1361_mixed_ipca_y','pY1361_mixed_ipca_z']]) 
        x_ipca= ClusteringAnalysis.ipca(x,2 )
        df_ipca = pd.DataFrame( x_ipca )
        df_ipca['y'] = y_predict
        plt.scatter(df_ipca[df_ipca['y']==1][0],df_ipca[df_ipca['y']==1][1],label='cluster positive',color='red')
        plt.scatter(df_ipca[df_ipca['y']==0][0],df_ipca[df_ipca['y']==0][1],label='cluster negative',color='blue')
        #plt.title('OCSVM Clustering')
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.legend(loc='best')
        fig_1=plt.gcf()
        png_name_roc='output_'+model_name+'_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name_roc)
        plt.close()       




        df_0 = df_similarity[(df_similarity['CodeC']==192) & (df_similarity['CodeB']==247) & (df_similarity['CodeA']==169)]
        df_1 = df_similarity[(df_similarity['CodeC']==192) & (df_similarity['CodeB']==245) & (df_similarity['CodeA']==191)]
        df_2 = df_similarity[(df_similarity['CodeC']==192) & (df_similarity['CodeB']==245) & (df_similarity['CodeA']==235)]
        df_3 = df_similarity[(df_similarity['CodeC']==192) & (df_similarity['CodeB']==207) & (df_similarity['CodeA']==134)]
        df_4 = df_similarity[(df_similarity['CodeC']==192) & (df_similarity['CodeB']==152) & (df_similarity['CodeA']==173)]
        df_5 = df_similarity[(df_similarity['CodeC']==192) & (df_similarity['CodeB']==247) & (df_similarity['CodeA']==174)]
        df_6 = df_similarity[(df_similarity['CodeC']==192) & (df_similarity['CodeB']==137) & (df_similarity['CodeA']==233)]
        df_7 = df_similarity[(df_similarity['CodeC']==1) & (df_similarity['CodeB']==17) & (df_similarity['CodeA']==71)]
        df_8 =df_similarity[(df_similarity['CodeC']==192) & (df_similarity['CodeB']==194) & (df_similarity['CodeA']==202)]
        df_9 =df_similarity[(df_similarity['CodeC']==192) & (df_similarity['CodeB']==194) & (df_similarity['CodeA']==52)]
        df_10 =df_similarity[(df_similarity['CodeC']==192) & (df_similarity['CodeB']==234) & (df_similarity['CodeA']==173)]
        df_11 =df_similarity[(df_similarity['CodeC']==192) & (df_similarity['CodeB']==245) & (df_similarity['CodeA']==156)]
        df_12 =df_similarity[(df_similarity['CodeC']==192) & (df_similarity['CodeB']==4) & (df_similarity['CodeA']==56)]
        df_13 =df_similarity[(df_similarity['CodeC']==192) & (df_similarity['CodeB']==42) & (df_similarity['CodeA']==166)]
        df_14 =df_similarity[(df_similarity['CodeC']==192) & (df_similarity['CodeB']==52) & (df_similarity['CodeA']==159)]
        df_15 =df_similarity[(df_similarity['CodeC']==260) & (df_similarity['CodeB']==26) & (df_similarity['CodeA']==22)]
        df_16 = df_similarity[(df_similarity['CodeC']==89) & (df_similarity['CodeB']==5) & (df_similarity['CodeA']==186)]

        df_sample = pd.concat((df_0,df_1,df_2,df_3,df_4,df_5),axis=0)

        df_sample = df_sample.reset_index()
        print('-----------samples are----------------')
        print(df_sample)
        text_sample=df_sample['CodeA'].astype(str)+'-'+df_sample['CodeB'].astype(str)+'-'+df_sample['CodeC'].astype(str)
        

        

        x_sample_1355 = df_sample['total_insr_pY1355_mixed_ipca']
        y_sample_1355 = df_sample['total_pY1355_mixed_ipca']
        print(df_sample)
        plt.scatter(x_ipca_1355[:,0],x_ipca_1355[:,1], alpha=0.3)
        '''
        for i, (comp,var) in enumerate(zip(ipca_1355.components_,ipca_1355.explained_variance_)):
            comp = comp * var
            print(comp)
            plt.plot([0,comp[0]],[0,comp[1]],label=f'Component{i}',linewidth=5,color=f'C{i+2}')
        '''
        #plt.title('Clustering')
        plt.plot(x_sample_1355,y_sample_1355,'r8')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.xlim(-10,10)
        plt.ylim(-1,5)
        for i in range(len(x_sample_1355)):
            plt.text(x_sample_1355[i],y_sample_1355[i]+0.2,text_sample[i],fontsize='x-small')
        plt.legend(loc='best')
        fig_2=plt.gcf()
        png_name_roc='output_ipca_pY1355_mixed_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name_roc)
        plt.close()  

        #x_sample = df_sample['total_insr']
        x_sample_1361 = df_sample['total_insr_pY1361_mixed_ipca']
        y_sample_1361 = df_sample['total_pY1361_mixed_ipca']
        plt.scatter(x_ipca_1361[:,0],x_ipca_1361[:,1], alpha=0.3)
        '''
        for i, (comp,var) in enumerate(zip(ipca_1361.components_,ipca_1361.explained_variance_)):
            comp = comp * var
            plt.plot([0,comp[0]],[0,comp[1]],label=f'Component{i}',linewidth=5,color=f'C{i+2}')
        '''
        #plt.title('Clustering')
        plt.plot(x_sample_1361,y_sample_1361,'r8')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.xlim(-10,10)
        plt.ylim(-1,5)
        for i in range(len(x_sample_1361)):
            plt.text(x_sample_1361[i],y_sample_1361[i]+0.2,text_sample[i],fontsize='x-small')
        plt.legend(loc='best')
        fig_3=plt.gcf()
        png_name_roc='output_ipca_pY1361_mixed_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name_roc)
        plt.close()  
        
        x_sample = df_sample['total_insr_pY1355_mixed']
        y_sample_1355 = df_sample['total_pY1355_mixed']
        plt.scatter(x_erh_1355[:,0],x_erh_1355[:,1], alpha=0.3)
        points = [(x,x) for x in range(0,4000)]
        x_line,y_line = zip(*points)
        plt.plot(x_line,y_line)
        plt.xlim(0,4000)
        plt.ylim(0,4000)
        plt.plot(x_sample,y_sample_1355,'r8')
        #plt.title('Clustering')
        plt.xlabel('Enrichment on Anti-INSR')
        plt.ylabel('Enrichment on Anti- pY1355')
        for i in range(len(x_sample)):
            plt.text(x_sample[i],y_sample_1355[i]+0.2,text_sample[i],fontsize='x-small')
        plt.legend(loc='best')
        fig_4=plt.gcf()
        png_name_roc='output_ipca_pY1355_orig_mixed_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name_roc)
        plt.close()  

        x_sample = df_sample['total_insr_pY1361_mixed']
        y_sample_1361 = df_sample['total_pY1361_mixed']
        plt.scatter(x_erh_1361[:,0],x_erh_1361[:,1], alpha=0.3)
        points = [(x,x) for x in range(0,2000)]
        x_line,y_line = zip(*points)
        plt.plot(x_line,y_line)
        #plt.title('Clustering')
        plt.xlim(0,2000)
        plt.ylim(0,2000)
        plt.xlabel('Enrichment on Anti-INSR')
        plt.ylabel('Enrichment on Anti-pY1361')
        plt.plot(x_sample,y_sample_1361,'r8')
        for i in range(len(x_sample)):
            plt.text(x_sample[i],y_sample_1361[i]+0.2,text_sample[i],fontsize='x-small')
        plt.legend(loc='best')
        fig_5=plt.gcf()
        png_name_roc='output_ipca_pY1361_orig_mixed_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name_roc)
        plt.close()  

        x_ipca_1355 = np.array(df_similarity[['total_insr_1355_ipca','total_pY1355_ipca']])
        x_ipca_1361 = np.array(df_similarity[['total_insr_1361_ipca','total_pY1361_ipca']])
        x_erh_1355 = np.array(df_similarity[['total_insr','total_pY1355']])    
        x_erh_1361 = np.array(df_similarity[['total_insr','total_pY1361']])   

        x_sample_1355 = df_sample['total_insr_1355_ipca']
        y_sample_1355 = df_sample['total_pY1355_ipca']
        print(df_sample)
        plt.scatter(x_ipca_1355[:,0],x_ipca_1355[:,1], alpha=0.3)
        '''
        for i, (comp,var) in enumerate(zip(ipca_1355.components_,ipca_1355.explained_variance_)):
            comp = comp * var
            print(comp)
            plt.plot([0,comp[0]],[0,comp[1]],label=f'Component{i}',linewidth=5,color=f'C{i+2}')
        '''
        #plt.title('Clustering')
        plt.plot(x_sample_1355,y_sample_1355,'r8')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        for i in range(len(x_sample_1355)):
            plt.text(x_sample_1355[i],y_sample_1355[i]+0.2,text_sample[i],fontsize='x-small')
        plt.legend(loc='best')
        fig_6=plt.gcf()
        png_name_roc='output_ipca_pY1355_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name_roc)
        plt.close()  

        #x_sample = df_sample['total_insr']
        x_sample_1361 = df_sample['total_insr_1361_ipca']
        y_sample_1361 = df_sample['total_pY1361_ipca']
        plt.scatter(x_ipca_1361[:,0],x_ipca_1361[:,1], alpha=0.3)
        '''
        for i, (comp,var) in enumerate(zip(ipca_1361.components_,ipca_1361.explained_variance_)):
            comp = comp * var
            plt.plot([0,comp[0]],[0,comp[1]],label=f'Component{i}',linewidth=5,color=f'C{i+2}')
        '''
        #plt.title('Clustering')
        plt.plot(x_sample_1361,y_sample_1361,'r8')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        for i in range(len(x_sample_1361)):
            plt.text(x_sample_1361[i],y_sample_1361[i]+0.2,text_sample[i],fontsize='x-small')
        plt.legend(loc='best')
        fig_7=plt.gcf()
        png_name_roc='output_ipca_pY1361_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name_roc)
        plt.close()  
        
        x_sample = df_sample['total_insr']
        y_sample_1355 = df_sample['total_pY1355']
        plt.scatter(x_erh_1355[:,0],x_erh_1355[:,1], alpha=0.3)
        points = [(x,x) for x in range(0,4000)]
        x_line,y_line = zip(*points)
        plt.plot(x_line,y_line)
        plt.xlim(0,4000)
        plt.ylim(0,4000)
        plt.plot(x_sample,y_sample_1355,'r8')
        #plt.title('Clustering')
        plt.xlabel('Enrichment on INSR')
        plt.ylabel('Enrichment on pY1355')
        for i in range(len(x_sample)):
            plt.text(x_sample[i],y_sample_1355[i]+0.2,text_sample[i],fontsize='x-small')
        plt.legend(loc='best')
        fig_8=plt.gcf()
        png_name_roc='output_ipca_pY1355_orig_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name_roc)
        plt.close()  

        x_sample = df_sample['total_insr']
        y_sample_1361 = df_sample['total_pY1361']
        plt.scatter(x_erh_1361[:,0],x_erh_1361[:,1], alpha=0.3)
        points = [(x,x) for x in range(0,4000)]
        x_line,y_line = zip(*points)
        plt.plot(x_line,y_line)
        #plt.title('Clustering')
        plt.xlim(0,4000)
        plt.ylim(0,4000)
        plt.xlabel('Enrichment on INSR')
        plt.ylabel('Enrichment on pY1355')
        plt.plot(x_sample,y_sample_1361,'r8')
        for i in range(len(x_sample)):
            plt.text(x_sample[i],y_sample_1361[i]+0.2,text_sample[i],fontsize='x-small')
        plt.legend(loc='best')
        fig_9=plt.gcf()
        png_name_roc='output_ipca_pY1361_orig_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name_roc)
        plt.close()  

        x_sample = df_sample['pY1355_ipca_x']
        y_sample = df_sample['pY1355_ipca_y']
        z_sample = df_sample['pY1355_ipca_z']
        ax = plt.axes(projection='3d')
        ax.scatter3D(x_antiinsr_pY1355_ipca[:,0],x_antiinsr_pY1355_ipca[:,1],x_antiinsr_pY1355_ipca[:,2], color='red',alpha=0.3)
        ax.plot(x_sample,y_sample,z_sample,'b8')
        #plt.title('Clustering')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_zlabel('PCA 3')
        for i in range(len(x_sample)):
            ax.text(x_sample[i],y_sample[i]+0.2,z_sample[i],text_sample[i],fontsize='x-small')
        plt.legend(loc='best')
        fig_10=plt.gcf()
        png_name_roc='output_ipca_antiinsr_pY1355_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name_roc)
        plt.close()  

        x_sample = df_sample['pY1361_ipca_x']
        y_sample = df_sample['pY1361_ipca_y']
        z_sample = df_sample['pY1361_ipca_z']
        ax = plt.axes(projection='3d')
        ax.scatter3D(x_antiinsr_pY1361_ipca[:,0],x_antiinsr_pY1361_ipca[:,1],x_antiinsr_pY1361_ipca[:,2], color='red',  alpha=0.3)
        ax.plot(x_sample,y_sample,z_sample,'b8')
        #plt.title('Clustering')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_zlabel('PCA 3')
        for i in range(len(x_sample)):
            ax.text(x_sample[i],y_sample[i]+0.2,z_sample[i],text_sample[i],fontsize='x-small')
        plt.legend(loc='best')
        fig_11=plt.gcf()
        png_name_roc='output_ipca_antiinsr_pY1361_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name_roc)
        plt.close()  

        x_sample = df_sample['pY1355_mixed_ipca_x']
        y_sample = df_sample['pY1355_mixed_ipca_y']
        z_sample = df_sample['pY1355_mixed_ipca_z']
        ax = plt.axes(projection='3d')
        ax.scatter3D( x_antiinsr_pY1355_ipca_mixed [:,0],x_antiinsr_pY1355_ipca_mixed [:,1],x_antiinsr_pY1355_ipca_mixed [:,2], color='red', alpha=0.3)
        #plt.title('Clustering')
        ax.plot(x_sample,y_sample,z_sample,'b8')
        #plt.title('Clustering')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_zlabel('PCA 3')
        for i in range(len(x_sample)):
            ax.text(x_sample[i],y_sample[i]+0.2,z_sample[i],text_sample[i],fontsize='x-small')
        plt.legend(loc='best')
        fig_12=plt.gcf()
        png_name_roc='output_ipca_antiinsr_pY1355_mixed_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name_roc)
        plt.close()  

        x_sample = df_sample['pY1361_mixed_ipca_x']
        y_sample = df_sample['pY1361_mixed_ipca_y']
        z_sample = df_sample['pY1361_mixed_ipca_z']
        ax = plt.axes(projection='3d')
        ax.scatter3D( x_antiinsr_pY1361_ipca_mixed [:,0],x_antiinsr_pY1361_ipca_mixed [:,1],x_antiinsr_pY1361_ipca_mixed [:,2], color='red',  alpha=0.3)
        #plt.title('Clustering')
        ax.plot(x_sample,y_sample,z_sample,'b8')
        #plt.title('Clustering')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_zlabel('PCA 3')
        for i in range(len(x_sample)):
            ax.text(x_sample[i],y_sample[i]+0.2,z_sample[i],text_sample[i],fontsize='x-small')
        plt.legend(loc='best')
        fig_13=plt.gcf()
        png_name_roc='output_ipca_antiinsr_pY1361_mixed_'+str(currentTime)+'.png'
        plt.savefig(graph_dir+png_name_roc)
        plt.close()  
        return fig_1,fig_2,fig_3,fig_4,fig_5,fig_6,fig_7,fig_8,fig_9,fig_10,fig_11



    def score(df_similarity):
        gc.collect()
        model_names = ['kmeans','AgglmerativeClustering']

        print(df_similarity)


        #df_similarity['Similarity_pY1355']=1/abs( df_similarity['total_insr_pY1355_mixed_ipca']-x_min_1355)
        #df_similarity['Similarity_pY1361']=1/abs( df_similarity['total_insr_pY1361_mixed_ipca']-x_min_1361)

        
        print('-------------------filtered dataframe is:------------------------')
        print(df_similarity)
        print(len(df_similarity))

        print(df_similarity [(df_similarity ['CodeC']==192) & (df_similarity ['CodeB']==247) & (df_similarity ['CodeA']==169)])
        print(df_similarity [(df_similarity ['CodeC']==192) & (df_similarity ['CodeB']==245) & (df_similarity ['CodeA']==191)])
        print(df_similarity [(df_similarity ['CodeC']==192) & (df_similarity ['CodeB']==245) & (df_similarity ['CodeA']==235)])
        print(df_similarity [(df_similarity ['CodeC']==192) & (df_similarity ['CodeB']==207) & (df_similarity ['CodeA']==134)])
        print(df_similarity [(df_similarity ['CodeC']==192) & (df_similarity ['CodeB']==152) & (df_similarity ['CodeA']==173)])
        print(df_similarity [(df_similarity ['CodeC']==192) & (df_similarity ['CodeB']==247) & (df_similarity ['CodeA']==174)])
        print(df_similarity [(df_similarity ['CodeC']==192) & (df_similarity ['CodeB']==137) & (df_similarity ['CodeA']==233)])
        print(df_similarity [(df_similarity ['CodeC']==1) & (df_similarity ['CodeB']==17) & (df_similarity ['CodeA']==71)])
        print(df_similarity [(df_similarity ['CodeC']==192) & (df_similarity ['CodeB']==137) & (df_similarity ['CodeA']==233)])
        print(df_similarity [(df_similarity ['CodeC']==192) & (df_similarity ['CodeB']==194) & (df_similarity ['CodeA']==202)])
        print(df_similarity [(df_similarity ['CodeC']==192) & (df_similarity ['CodeB']==194) & (df_similarity ['CodeA']==52)])
        print(df_similarity [(df_similarity ['CodeC']==192) & (df_similarity ['CodeB']==234) & (df_similarity ['CodeA']==173)])        
        print(df_similarity [(df_similarity ['CodeC']==192) & (df_similarity ['CodeB']==245) & (df_similarity ['CodeA']==156)])  
        
        #gc.collect()

        #df_similarity = pd.read_csv('/Users/xiaotan/Documents/chemistry/data/phase_1/result_kmeans_insr.csv')

        #df_similarity['performance_ind']=max(df_similarity['performance_ind_0_total'],df_similarity['performance_ind_1_total'])
        #df_similarity ['Similarity'] = (df_similarity ['Similarity'].pow(max(df_similarity['performance_ind_0_total'],df_similarity['performance_ind_1_total']).apply(np.exp)))*df_similarity ['S1_Richness_efficiency']*df_similarity ['S1_Richness_balance']
        #df_similarity ['Similarity'] = df_similarity ['Similarity']*df_similarity ['S1_Richness_balance']
        #df_similarity_weighted = df_similarity[df_similarity['S1_Richness_efficiency']>average_efficiency]
        #df_similarity ['Similarity'] = (df_similarity ['Similarity']*abs(df_similarity ['Richness_STDEV']+df_similarity ['S1_STDEV']))
        

        #df_similarity_0['SCORE_RANK']=df_similarity_0['Score'].rank(ascending=True).astype(float)
        df_similarity_1 = df_similarity[df_similarity['class']==1]
        df_similarity_1= df_similarity_1.sort_values(by =['Similarity'],ascending=True)
        df_similarity_1['Similarity_antiinsr_RANK_pY1355']=df_similarity_1['Similarity_antiinsr_1355'].rank(ascending=True).astype(float)
        df_similarity_1['Similarity_antiinsr_RANK_pY1361']=df_similarity_1['Similarity_antiinsr_1361'].rank(ascending=True).astype(float)
        df_similarity_1['Similarity_RANK_pY1355']=df_similarity_1['Similarity_pY1355'].rank(ascending=True).astype(float)
        df_similarity_1['Similarity_RANK_pY1361']=df_similarity_1['Similarity_pY1361'].rank(ascending=True).astype(float)
        df_similarity_1['Similarity_centralLine_RANK_pY1355']=df_similarity_1['Similarity_centralLine_pY1355'].rank(ascending=True).astype(float)
        df_similarity_1['Similarity_centralLine_RANK_pY1361']=df_similarity_1['Similarity_centralLine_pY1361'].rank(ascending=True).astype(float)
        df_similarity_1['Similarity_point_RANK_pY1355']=df_similarity_1['Similarity_point_pY1355'].rank(ascending=True).astype(float)
        df_similarity_1['Similarity_point_RANK_pY1361']=df_similarity_1['Similarity_point_pY1361'].rank(ascending=True).astype(float)
        df_similarity_1['Similarity_point_RANK_pY1355_mixed']=df_similarity_1['Similarity_point_pY1355_mixed'].rank(ascending=True).astype(float)
        df_similarity_1['Similarity_point_RANK_pY1361_mixed']=df_similarity_1['Similarity_point_pY1361_mixed'].rank(ascending=True).astype(float)
        df_similarity_1['S1_Richness_efficiency_RANK']=df_similarity_1['S1_Richness_efficiency'].rank(ascending=False).astype(float)
        df_similarity_1['S1_Richness_balance_RANK']=df_similarity_1['S1_Richness_balance'].rank(ascending=False).astype(float)
        if model_name in model_names:
            df_similarity_1['Similarity_RANK']=df_similarity_1['Similarity'].rank(ascending=False).astype(float)
        else:
            df_similarity_1['Similarity_RANK']=df_similarity_1['Similarity'].rank(ascending=True).astype(float)
        df_similarity_1['Richness_SUM_RANK']=df_similarity_1['Richness_SUM'].rank(ascending=False).astype(float)
        df_similarity_1['Richness_COUNT_RANK']=df_similarity_1['Richness_COUNT'].rank(ascending=False).astype(float)
        df_similarity_1['S1_SUM_RANK']=df_similarity_1['S1_SUM'].rank(ascending=False).astype(float)
        df_similarity_1['S1_COUNT_RANK']=df_similarity_1['S1_COUNT'].rank(ascending=False).astype(float)
        #df_similarity_1['Score']=(df_similarity_1['Similarity_RANK']+(df_similarity_1['Similarity_point_RANK_pY1355_mixed']+df_similarity_1['Similarity_point_RANK_pY1361_mixed'])/2).astype(float)
        df_similarity_1['Score']=df_similarity_1['Similarity_RANK']+(df_similarity_1['Richness_COUNT_RANK']+df_similarity_1['Richness_SUM_RANK'])/2
        #df_similarity_1['Score']=(df_similarity_1['Similarity_RANK']+(df_similarity_1['Similarity_antiinsr_RANK_pY1355']+df_similarity_1['Similarity_antiinsr_RANK_pY1361'])).astype(float)
        #df_similarity_1['Score']=df_similarity_1['Similarity_RANK']+(df_similarity_1['Richness_COUNT_RANK']+df_similarity_1['Richness_SUM_RANK'])
        #df_similarity_1['Score']=(df_similarity_1['Similarity_RANK']*0.5+(df_similarity_1['Richness_COUNT_RANK']+df_similarity_1['Richness_SUM_RANK'])/2 +(df_similarity_1['Similarity_RANK_pY1355']+df_similarity_1['Similarity_RANK_pY1361'])/2  ).astype(float)
        #df_similarity_1['Score']=(df_similarity_1['Similarity_RANK']+(df_similarity_1['S1_Richness_balance_RANK']+df_similarity_1['S1_Richness_efficiency_RANK'])*0.5).astype(float)
        #df_similarity_1['Score']=df_similarity_1['Similarity_centralLine']+df_similarity_1['Similarity_RANK']+df_similarity_1['Richness_COUNT_RANK']+df_similarity_1['Richness_SUM_RANK']
        #df_similarity_1['Score']=df_similarity_1['Similarity_RANK_pY1355']+df_similarity_1['Similarity_RANK_pY1361']+df_similarity_1['Similarity_RANK']+(df_similarity_1['Richness_COUNT_RANK']+df_similarity_1['Richness_SUM_RANK'])/2
        #df_similarity_1['Score']=df_similarity_1['Similarity_RANK_pY1355']+df_similarity_1['Similarity_RANK_pY1361']+(df_similarity_1['Richness_COUNT_RANK']+df_similarity_1['Richness_SUM_RANK'])/2        
        '''
        top_efficiency = df_similarity_1.sort_values(by =['S1_Richness_efficiency'],ascending=True).head(100)
        top_balance = df_similarity_1.sort_values(by =['S1_Richness_balance'],ascending=True).head(100)
        for i in range(len(df_similarity_1)):
            for j in range(len(top_efficiency)):
                #print(top_efficiency['S1_Richness_efficiency'].values[j], top_balance['S1_Richness_balance'].values[j])
                if df_similarity_1['S1_Richness_efficiency'].values[i] == top_efficiency['S1_Richness_efficiency'].values[j]:
                    df_similarity_1['Score'].values[i] = df_similarity_1['Score'].values[i] + df_similarity_1['S1_Richness_efficiency_RANK'].values[i]
                elif df_similarity_1['S1_Richness_balance'].values[i] == top_balance['S1_Richness_balance'].values[j]:
                    df_similarity_1['Score'].values[i]=df_similarity_1['Score'].values[i] + df_similarity_1['S1_Richness_balance_RANK'].values[i]
        '''
        for i in range(len(df_similarity_1)):
            weight = max(df_similarity_1['performance_ind_0_total'].values[i],df_similarity_1['performance_ind_1_total'].values[i])
            #print(np.exp(max(weight,1)),weight)
            #df_similarity_1 ['Score'].values[i] = np.power(df_similarity_1 ['Score'].values[i],max(weight,1))
            
            
            if weight==5:
                df_similarity_1 ['Score'].values[i] = df_similarity_1 ['Score'].values[i]/160
            elif weight==4:
                df_similarity_1 ['Score'].values[i] = df_similarity_1 ['Score'].values[i]/144
            elif weight==3:
                df_similarity_1 ['Score'].values[i] = df_similarity_1 ['Score'].values[i]/32
            elif weight==2:
                df_similarity_1 ['Score'].values[i] = df_similarity_1 ['Score'].values[i]/4
            elif weight==1:
                df_similarity_1 ['Score'].values[i] = df_similarity_1 ['Score'].values[i]/2    


        df_similarity_1['SCORE_RANK']=df_similarity_1['Score'].rank(ascending=True).astype(float)

        df_similarity_0 = df_similarity[df_similarity['class']==0]
        df_similarity_0= df_similarity_0.sort_values(by =['Similarity'],ascending=True)
        df_similarity_0['Similarity_RANK_pY1355']=df_similarity_0['Similarity_pY1355'].rank(ascending=True).astype(float)
        df_similarity_0['Similarity_RANK_pY1355']=df_similarity_0['Similarity_pY1355'].rank(ascending=True).astype(float)
        df_similarity_0['Similarity_RANK_pY1361']=df_similarity_0['Similarity_pY1361'].rank(ascending=True).astype(float)
        df_similarity_0['Similarity_centralLine_RANK_pY1355']=df_similarity_0['Similarity_centralLine_pY1355'].rank(ascending=True).astype(float)        
        df_similarity_0['Similarity_centralLine_RANK_pY1361']=df_similarity_0['Similarity_centralLine_pY1361'].rank(ascending=True).astype(float)
        df_similarity_0['Similarity_point_RANK_pY1355']=df_similarity_0['Similarity_point_pY1355'].rank(ascending=True).astype(float)
        df_similarity_0['Similarity_point_RANK_pY1361']=df_similarity_0['Similarity_point_pY1361'].rank(ascending=True).astype(float)
        df_similarity_0['Similarity_point_RANK_pY1355_mixed']=df_similarity_0['Similarity_point_pY1355_mixed'].rank(ascending=True).astype(float)
        df_similarity_0['Similarity_point_RANK_pY1361_mixed']=df_similarity_0['Similarity_point_pY1361_mixed'].rank(ascending=True).astype(float)
        df_similarity_0['Similarity_antiinsr_RANK_pY1355']=df_similarity_0['Similarity_antiinsr_1355'].rank(ascending=True).astype(float)
        df_similarity_0['Similarity_antiinsr_RANK_pY1361']=df_similarity_0['Similarity_antiinsr_1361'].rank(ascending=True).astype(float)
        df_similarity_0['S1_Richness_efficiency_RANK']=df_similarity_0['S1_Richness_efficiency'].rank(ascending=False).astype(float)
        df_similarity_0['S1_Richness_balance_RANK']=df_similarity_0['S1_Richness_balance'].rank(ascending=False).astype(float)
        if model_name in model_names:
            df_similarity_0['Similarity_RANK']=df_similarity_0['Similarity'].rank(ascending=False).astype(float)
        else:
            df_similarity_0['Similarity_RANK']=df_similarity_0['Similarity'].rank(ascending=True).astype(float)
        df_similarity_0['Richness_SUM_RANK']=df_similarity_0['Richness_SUM'].rank(ascending=False).astype(float)
        df_similarity_0['Richness_COUNT_RANK']=df_similarity_0['Richness_COUNT'].rank(ascending=False).astype(float)
        df_similarity_0['S1_SUM_RANK']=df_similarity_0['S1_SUM'].rank(ascending=False).astype(float)
        df_similarity_0['S1_COUNT_RANK']=df_similarity_0['S1_COUNT'].rank(ascending=False).astype(float)
        #df_similarity_0['Score']=(df_similarity_0['Similarity_RANK']+(df_similarity_0['Similarity_point_RANK_pY1355_mixed']+df_similarity_0['Similarity_point_RANK_pY1361_mixed'])/2).astype(float)
        #df_similarity_0['Score']=(df_similarity_0['Similarity_RANK']+(df_similarity_0['Similarity_antiinsr_RANK_pY1355']+df_similarity_0['Similarity_antiinsr_RANK_pY1361'])).astype(float)
        #df_similarity_0['Score']=df_similarity_0['Similarity_RANK']+(df_similarity_0['Richness_COUNT_RANK']+df_similarity_0['Richness_SUM_RANK'])
        df_similarity_0['Score']=df_similarity_0['Similarity_RANK']+(df_similarity_0['Richness_COUNT_RANK']+df_similarity_0['Richness_SUM_RANK'])/2
        #df_similarity_0['Score']=df_similarity_0['Similarity_centralLine_RANK']+df_similarity_0['Similarity_RANK']+(df_similarity_0['Richness_COUNT_RANK']+df_similarity_0['Richness_SUM_RANK'])/2
        #df_similarity_0['Score']=(df_similarity_0['Similarity_RANK']+(df_similarity_0['S1_Richness_balance_RANK']+df_similarity_0['S1_Richness_efficiency_RANK'])*0.5).astype(float)
        #df_similarity_0['Score']=((df_similarity_0['Similarity_RANK_pY1355']+df_similarity_0['Similarity_RANK_pY1361'])*0.5+df_similarity_0['Similarity_RANK']).astype(float)       
        #df_similarity_0['Score']=(df_similarity_0['Similarity_RANK']*0.5+(df_similarity_0['Richness_COUNT_RANK']+df_similarity_0['Richness_SUM_RANK'])/2+(df_similarity_0['Similarity_RANK_pY1355']+df_similarity_0['Similarity_RANK_pY1361'])/2).astype(float)
        #df_similarity_0['Score']=df_similarity_0['Similarity_centralLine']+df_similarity_0['Similarity_RANK']+df_similarity_0['Richness_COUNT_RANK']+df_similarity_0['Richness_SUM_RANK']
        #df_similarity_0['Score']=df_similarity_0['Similarity_RANK_pY1355']+df_similarity_0['Similarity_RANK_pY1361']+(df_similarity_0['Richness_COUNT_RANK']+df_similarity_0['Richness_SUM_RANK'])/2
        '''
        top_efficiency = df_similarity_0.sort_values(by =['S1_Richness_efficiency'],ascending=True).head(100)
        top_balance = df_similarity_0.sort_values(by =['S1_Richness_balance'],ascending=True).head(100)
        for i in range(len(df_similarity_0)):
            for j in range(len(top_efficiency)):
                #print(top_efficiency['S1_Richness_efficiency'].values[j], top_balance['S1_Richness_balance'].values[j])
                if df_similarity_0['S1_Richness_efficiency'].values[i] == top_efficiency['S1_Richness_efficiency'].values[j]:
                    df_similarity_0['Score'].values[i] = df_similarity_0['Score'].values[i] + df_similarity_0['S1_Richness_efficiency_RANK'].values[i]
                elif df_similarity_0['S1_Richness_balance'].values[i] == top_balance['S1_Richness_balance'].values[j]:
                    df_similarity_0['Score'].values[i]=df_similarity_0['Score'].values[i] + df_similarity_0['S1_Richness_balance_RANK'].values[i]
        '''
        for i in range(len(df_similarity_0)):
            
            weight = max(df_similarity_0['performance_ind_0_total'].values[i],df_similarity_0['performance_ind_1_total'].values[i])
            #print(np.exp(max(weight,1)),weight)
            #df_similarity_0 ['Score'].values[i] = np.power(df_similarity_0 ['Score'].values[i],max(weight,1))
            
            if weight==5:
                df_similarity_0 ['Score'].values[i] = df_similarity_0 ['Score'].values[i]/160
            elif weight==4:
                df_similarity_0 ['Score'].values[i] = df_similarity_0 ['Score'].values[i]/144
            elif weight==3:
                df_similarity_0 ['Score'].values[i] = df_similarity_0 ['Score'].values[i]/32
            elif weight==2:
                df_similarity_0 ['Score'].values[i] = df_similarity_0 ['Score'].values[i]/4
            elif weight==1:
                df_similarity_0 ['Score'].values[i] = df_similarity_0 ['Score'].values[i]/2
                    


        df_similarity_0['SCORE_RANK']=df_similarity_0['Score'].rank(ascending=True).astype(float)+len(df_similarity_1)
        df_score_ = pd.concat((df_similarity_0,df_similarity_1),axis=0)
        df_rank = pd.DataFrame()
        df_rank['SCORE_RANK'] = df_score_['SCORE_RANK'].drop_duplicates()
        df_rank['RANK'] = df_rank['SCORE_RANK'].rank(ascending=True).astype(float)
        df_score = df_score_.merge(df_rank,how='left',on=['SCORE_RANK'])
        df_score['SCORE_RANK']=df_score['SCORE_RANK'].rank(ascending=True).astype(float)
        df_score= df_score.sort_values(by =['SCORE_RANK'],ascending=True)
        print('-------------------filtered dataframe is:------------------------')
        print(df_score)
        print(len(df_score))
        df_score=df_score.reset_index()
        print(df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==247) & (df_score ['CodeA']==169)])
        print(df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==245) & (df_score ['CodeA']==191)])
        print(df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==245) & (df_score ['CodeA']==235)])
        print(df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==207) & (df_score ['CodeA']==134)])
        print(df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==152) & (df_score ['CodeA']==173)])
        print(df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==247) & (df_score ['CodeA']==174)])
        print(df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==137) & (df_score ['CodeA']==233)])
        print(df_score [(df_score ['CodeC']==1) & (df_score ['CodeB']==17) & (df_score ['CodeA']==71)])
        print(df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==137) & (df_score ['CodeA']==233)])
        print(df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==194) & (df_score ['CodeA']==202)])
        print(df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==194) & (df_score ['CodeA']==52)])
        print(df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==234) & (df_score ['CodeA']==173)])        
        print(df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==245) & (df_score ['CodeA']==156)])  
        print(df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==248) & (df_score ['CodeA']==260)])
        print(df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==218) & (df_score ['CodeA']==40)])
        print(df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==164) & (df_score ['CodeA']==235)])
        print(df_score [(df_score ['CodeC']==340) & (df_score ['CodeB']==114) & (df_score ['CodeA']==15)])
        
        df_0 = df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==247) & (df_score ['CodeA']==169)]
        df_1 = df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==245) & (df_score ['CodeA']==191)]
        df_2 = df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==245) & (df_score ['CodeA']==235)]
        df_3 = df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==207) & (df_score ['CodeA']==134)]
        df_4 = df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==152) & (df_score ['CodeA']==173)]
        df_5 = df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==247) & (df_score ['CodeA']==174)]
        df_6 = df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==137) & (df_score ['CodeA']==233)]
        df_7 = df_score [(df_score ['CodeC']==1) & (df_score ['CodeB']==17) & (df_score ['CodeA']==71)]
        df_8 =df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==194) & (df_score ['CodeA']==202)]
        df_9 =df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==194) & (df_score ['CodeA']==52)]
        df_10 =df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==234) & (df_score ['CodeA']==173)]
        df_11 =df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==245) & (df_score ['CodeA']==156)]
        df_12 =df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==169) & (df_score ['CodeA']==252)]
        df_13 =df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==56) & (df_score ['CodeA']==173)]
        df_14 =df_score [(df_score ['CodeC']==192) & (df_score ['CodeB']==118) & (df_score ['CodeA']==205)]
        df_15 =df_score [(df_score ['CodeC']==140) & (df_score ['CodeB']==197) & (df_score ['CodeA']==14)]
        df_sample = pd.concat((df_0,df_1,df_2,df_3,df_4,df_5,df_6,df_7,df_9,df_10,df_11,df_12,df_13,df_14,df_15),axis=0)
        df_sample.to_csv('sample_20241017.csv')
        #print(dist)
        result_name = 'output_'+model_name+'_level1.csv'
        df_score.to_csv(output_dir+result_name)

        return df_score


if __name__=='__main__':
    
    data_dir = 'C:/Users/sharo/Documents/chemistry/chemistry_phase1/chemistry_phase1/data/phase_1/rs/total.csv'
    temp_dir = 'C:/Users/sharo/Documents/chemistry/chemistry_phase1/chemistry_phase1/parameters/phase_1/level_0/'
    erh_dir = 'C:/Users/sharo/Documents/chemistry/chemistry_phase1/chemistry_phase1/data/phase_1/erh/'
    graph_dir='C:/Users/sharo/Documents/chemistry/chemistry_phase1/chemistry_phase1/graph/phase_1/level_0/'
    output_dir='C:/Users/sharo/Documents/chemistry/chemistry_phase1/chemistry_phase1/output/phase_1/level_0/'
    #gc.collect()
    gc.collect()
    df_orig = dataLoading(data_dir)
    #x,x_dual = indExtract(df)

    df_normalized = preprocess.dataNormalize(df_orig)
    df = preprocess.dataPreprocess(df_normalized)



    x_efficiency = np.array(df['S1_Richness_efficiency'])
    x_efficiency = np.reshape(x_efficiency,(len(x_efficiency),1))  
    df_filtered = preprocess.outlierFiltering(x_efficiency,df,1)
    x_balance= np.array(df_filtered['S1_Richness_balance'])
    x_balance = np.reshape(x_balance,(len(x_balance),1))
    df_filtered = preprocess.outlierFiltering(x_balance,df_filtered,2)
    #x_dual = np.array(df_filtered[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']])
    #df_filtered = outlierFiltering(x_dual,df_filtered,3)
    #x_bind_0 = np.array(df_filtered[['S1_ind','Richness_ind','S1_Richness_balance','S1_Richness_efficiency']])
    #df_filtered = outlierFiltering(x_bind_0,df_filtered,1)
    x_bind_1 = np.array(df_filtered[['S1_ind','Richness_ind']])
    df_filtered = preprocess.outlierFiltering(x_bind_1,df_filtered,3)
    #x_bind_2 = np.array(df_filtered[['S1_STDEV','Richness_STDEV']])
    #df_filtered = preprocess.outlierFiltering(x_bind_2,df_filtered,4)

    df_erh_insr = similarityAnalysis.erhAnalysis(erh_dir,df_filtered)
    y_predict,model_name,x,df_result =similarityAnalysis.classification(df_filtered)
    df_erh_insr_ipca = similarityAnalysis.ipcaAnalysis(df_erh_insr,df_result)
    
    df_similarity = similarityAnalysis.similarity(df_erh_insr_ipca,model_name )

    fig_1,fig_2,fig_3,fig_4,fig_5,fig_6,fig_7,fig_8,fig_9,fig_10,fig_11= evaluation.Visualize(x,df_similarity,y_predict)
    classes=set(y_predict)
    

    df_score = evaluation.score(df_similarity )
    del df_result
    del df_erh_insr
    del df_filtered
    del df_similarity
    del df

    
