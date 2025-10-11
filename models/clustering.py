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


def opticsClustering(df,opt,samples):
	gc.collect()
	output_dir = opt.output_dir

	x = np.array(df[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']])
	df_result = df
	ordering,core_distances,reachability,predecessor=compute_optics_graph(x,min_samples=2,max_eps=np.inf,metric='euclidean',p=2,metric_params=None,algorithm='auto',leaf_size=2,n_jobs=None)
	labels,clusters=cluster_optics_xi(reachability=reachability,predecessor=predecessor,ordering=ordering,min_samples=2,xi=0.00001)

	y_predict = labels
	df_result['class'] = y_predict
	if len(samples)>0:
		for i in range(len(samples)):
			codeA=samples[i][0]
			codeB=samples[i][1]
			codeC=samples[i][2]
			print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])
	df_result.to_csv(output_dir+'labels_opticsClustering.csv',chunksize=1000)

	result_name='output_'+opt.model_name+'.csv'
	df_result.to_csv(output_dir+result_name)
	return y_predict
def kmeans(df,opt,samples):
	gc.collect()
	output_dir = opt.output_dir

	x = np.array(df[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']])
	df_result = df
	model = KMeans(n_clusters=2,init='k-means++',algorithm='lloyd',tol=5e-4)

	model = model.fit(x)
	y_predict = model.predict(x)
	df_result['class'] = y_predict
	if len(samples)>0:
		for i in range(len(samples)):
			codeA=samples[i][0]
			codeB=samples[i][1]
			codeC=samples[i][2]
			print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])
	df_result.to_csv(output_dir+'labels_kmeans.csv',chunksize=1000)

	score_of_determination = model.score(x)

	print('Score of model is: ',score_of_determination)

	result_name='output_'+opt.model_name+'.csv'
	df_result.to_csv(output_dir+result_name)
	return y_predict




def agglomerativeClustering(df,opt,samples):
	gc.collect()
	output_dir = opt.output_dir
	x = np.array(df[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']])

	df_result = df

	model = AgglomerativeClustering(n_clusters=2,metric='euclidean',linkage='ward',compute_distances=True)

	model = model.fit(x)
	print('distance is: ',model.distances_)
	print(np.mean(model.distances_),max(model.distances_),model.n_clusters_)
	y_predict = model.labels_
	df_result['class'] = y_predict
	if len(samples)>0:
		for i in range(len(samples)):
			codeA=samples[i][0]
			codeB=samples[i][1]
			codeC=samples[i][2]
			print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])
	df_result.to_csv(output_dir+'labels_agglomerative.csv',chunksize=1000)

	result_name='output_'+opt.model_name+'.csv'
	df_result.to_csv(output_dir+result_name)

	return y_predict
def birch(df,opt,samples):
	gc.collect()
	output_dir = opt.output_dir

	x = np.array(df[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']])

	df_result = df

	model = Birch(threshold=0.7819798519436253,branching_factor=40,n_clusters=2)

	model = model.fit(x)
	y_predict = model.predict(x)

	df_result['class'] = y_predict
	if len(samples)>0:
		for i in range(len(samples)):
			codeA=samples[i][0]
			codeB=samples[i][1]
			codeC=samples[i][2]
			print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])

	result_name='output_'+opt.model_name+'.csv'
	df_result.to_csv(output_dir+result_name)

	return y_predict
def spectral(df,opt,samples):
	gc.collect()
	output_dir = opt.output_dir

	x = np.array(df[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']])
      
	df_result = df
	model = SpectralClustering(n_clusters=2,assign_labels='kmeans',eigen_solver='arpack',random_state=0,affinity='nearest_neighbors')

	model = model.fit(x)
	y_predict = model.labels_
	df_result['class'] = y_predict
	if len(samples)>0:
		for i in range(len(samples)):
			codeA=samples[i][0]
			codeB=samples[i][1]
			codeC=samples[i][2]
			print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])


	result_name='output_'+opt.model_name+'.csv'
	df_result.to_csv(output_dir+result_name)
	return y_predict


