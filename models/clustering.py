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
import matplotlib.pyplot as plt
import gc


def opticsClustering(df,opt):
	gc.collect()
	output_dir = opt.output_dir
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
def kmeans(df,opt):
	gc.collect()
	output_dir = opt.output_dir
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




def agglomerativeClustering(df,opt):
	gc.collect()
	output_dir = opt.output_dir
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
def birch(df,opt):
	gc.collect()
	output_dir = opt.output_dir
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
def spectral(df,opt):
	gc.collect()
	output_dir = opt.output_dir
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


