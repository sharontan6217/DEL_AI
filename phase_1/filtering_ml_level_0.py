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

import models
from models import ipca
import similarityAnalysis
from similarityAnalysis import ipcaAnalysis,similarity
from scipy import stats as st
import matplotlib.pyplot as plt
import torch
import utils
from utils import preprocess, utils
import argparse
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

    return df_orig


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
	y_predict,model_name,score_of_determination = models.oneclassSVM.oneclassSVM(x,opt,currentTime)
	#y_predict,model_name  = models.clustering.kmeans(df_filtered,opt)
	#y_predict,model_name = model.oneclassSVM(df_filtered,opt)
	#y_predict,model_name  = models.clustering.spectral(df_filtered,opt)

	#y_predict,model_name  = models.clustering.birch(df_filtered,opt)
	#y_predict,model_name  = models.clustering.agglomerativeClustering(df_filtered,opt)
	#y_predict,model_name  = models.clustering.opticsClustering(df_filtered,opt)
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
        x_ipca= ipca.ipca(x,2 )
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
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='./chemistry_phase1/chemistry_phase1/data/phase_1/rs/total.csv', help = 'directory of the original data' )
    parser.add_argument('--erh_dir',type=str,default='./chemistry_phase1/chemistry_phase1/data/phase_1/erh/', help = 'directory of erh files' )
    parser.add_argument('--graph_dir',type=str,default='./chemistry_phase1/chemistry_phase1/graph/phase_1/level_0/', help = 'directory of graphs' )
    parser.add_argument('--output_dir',type=str,default='./chemistry_phase1/chemistry_phase1/output/phase_1/level_0/', help = 'directory of outputs')
    opt = parser.parse_args()
    return opt

if __name__=='__main__':
    
    #data_dir = 'C:/Users/sharo/Documents/chemistry/chemistry_phase1/chemistry_phase1/data/phase_1/rs/total.csv'
    #erh_dir = 'C:/Users/sharo/Documents/chemistry/chemistry_phase1/chemistry_phase1/data/phase_1/erh/'
    #graph_dir='C:/Users/sharo/Documents/chemistry/chemistry_phase1/chemistry_phase1/graph/phase_1/level_0/'
    #output_dir='C:/Users/sharo/Documents/chemistry/chemistry_phase1/chemistry_phase1/output/phase_1/level_0/'
    #gc.collect()
    
    gc.collect()
    opt = get_parser()
    data_dir = opt.data_dir
    erh_dir = opt.erh_dir
    graph_dir = opt.graph_dir
    output_dir = opt.output_dir
    df_orig = dataLoading(data_dir)
    #x,x_dual = indExtract(df)
    df_orig = preprocess.descriptors(df_orig)
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

    df_erh_insr = erhAnalysis(erh_dir,df_filtered)
    y_predict,model_name,x,df_result =classification(df_filtered)
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

    
