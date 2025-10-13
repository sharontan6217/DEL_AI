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
from models import Config
from Config import opticsClustering_config,kmeans_config,agglomerativeClustering_config,birch_config,spectral_config
import gc


def opticsClustering(df,opt,samples):
	gc.collect()
	output_dir = opt.output_dir
	config = Config.opticsClustering_config()

	x = np.array(df[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']])
	df_result = df
	ordering,core_distances,reachability,predecessor=compute_optics_graph(x,**config)
	labels,clusters=cluster_optics_xi(reachability=reachability,predecessor=predecessor,ordering=ordering,min_samples=2,xi=0.00001)

	y_predict = labels
	df_result['class'] = y_predict
	if len(samples)>0:
		for i in range(len(samples)):
			codeA=samples['CodeA'][i]
			codeB=samples['CodeB'][i]
			codeC=samples['CodeC'][i]
			print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])
	df_result.to_csv(output_dir+'labels_opticsClustering.csv',chunksize=1000)

	result_name='output_'+opt.model_name+'.csv'
	df_result.to_csv(output_dir+result_name)
	return y_predict
def kmeans(df,opt,samples):
	gc.collect()
	config = Config.kmeans_config()
	output_dir = opt.output_dir

	x = np.array(df[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']])
	df_result = df
	model = KMeans(**config)

	model = model.fit(x)
	y_predict = model.predict(x)
	df_result['class'] = y_predict
	if len(samples)>0:
		for i in range(len(samples)):
			codeA=samples['CodeA'][i]
			codeB=samples['CodeB'][i]
			codeC=samples['CodeC'][i]
			print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])
	df_result.to_csv(output_dir+'labels_kmeans.csv',chunksize=1000)

	score_of_determination = model.score(x)

	print('Score of model is: ',score_of_determination)

	result_name='output_'+opt.model_name+'.csv'
	df_result.to_csv(output_dir+result_name)
	return y_predict




def agglomerativeClustering(df,opt,samples):
	gc.collect()
	config = Config.agglomerativeClustering_config()
	output_dir = opt.output_dir
	x = np.array(df[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']])

	df_result = df

	model = AgglomerativeClustering(**config)

	model = model.fit(x)
	print('distance is: ',model.distances_)
	print(np.mean(model.distances_),max(model.distances_),model.n_clusters_)
	y_predict = model.labels_
	df_result['class'] = y_predict
	if len(samples)>0:
		for i in range(len(samples)):
			codeA=samples['CodeA'][i]
			codeB=samples['CodeB'][i]
			codeC=samples['CodeC'][i]
			print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])
	df_result.to_csv(output_dir+'labels_agglomerative.csv',chunksize=1000)

	result_name='output_'+opt.model_name+'.csv'
	df_result.to_csv(output_dir+result_name)

	return y_predict
def birch(df,opt,samples):
	gc.collect()
	config = Config.birch_config()
	output_dir = opt.output_dir

	x = np.array(df[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']])

	df_result = df

	model = Birch(**config)

	model = model.fit(x)
	y_predict = model.predict(x)

	df_result['class'] = y_predict
	if len(samples)>0:
		for i in range(len(samples)):
			codeA=samples['CodeA'][i]
			codeB=samples['CodeB'][i]
			codeC=samples['CodeC'][i]
			print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])

	result_name='output_'+opt.model_name+'.csv'
	df_result.to_csv(output_dir+result_name)

	return y_predict
def spectral(df,opt,samples):
	gc.collect()
	config = Config.spectral_configg()
	output_dir = opt.output_dir

	x = np.array(df[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']])
      
	df_result = df
	model = SpectralClustering(**config)

	model = model.fit(x)
	y_predict = model.labels_
	df_result['class'] = y_predict
	if len(samples)>0:
		for i in range(len(samples)):
			codeA=samples['CodeA'][i]
			codeB=samples['CodeB'][i]
			codeC=samples['CodeC'][i]
			print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])


	result_name='output_'+opt.model_name+'.csv'
	df_result.to_csv(output_dir+result_name)
	return y_predict


