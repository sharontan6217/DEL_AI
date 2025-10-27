
import numpy as np
import pandas as pd
import random
import sklearn
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,ShuffleSplit,HalvingGridSearchCV
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.svm import OneClassSVM
import models
from models import oneclassSVM, Config
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, KernelDensity
from scipy.spatial.distance import sqeuclidean,jaccard,canberra,cdist,euclidean
def AEC(x):
	kde=KernelDensity(kernel='gaussian').fit(x)
	dens=kde.score_samples(x)          

	average_dens=np.mean(dens)
	
	print('average density is: ', average_dens)          
		
	if abs(average_dens)<2:
		min_samples_aec=2
	else:
		min_samples_aec=int(abs(average_dens))
	print("min_samples are: ", min_samples_aec)
	
	x_ = np.array(x)
	dist=[]
	for i in range (len(x)):

		dist_pair=cdist(np.array(np.reshape(x[i],(1,len(x[i])))),x_,'euclidean')
		average_dist_= np.mean(dist_pair)

		dist.append(average_dist_)

	print(len(dist))
	average_dist=max(dist)
	print('average distance is: ', average_dist)
	eps_aec=average_dist/(2*min_samples_aec)

	print("eps is: ", eps_aec)
		
	
	print("min_samples are: ", min_samples_aec)
	return eps_aec,min_samples_aec


def boundary(x,x_):
	similarity=[]
	for i in range (len(x)):
		similarity_=[]
		for j in range(len(x_)):
			#dist_pair=cdist(np.reshape(x[i],(len(x[i]),1)),np.reshape(x_[j],(len(x[j]),1)),'euclidean')
			dist=euclidean(x[i],x_[j])
			similarity_point = 1/dist
			similarity_.append(similarity_point)
		average_similarity=np.average(similarity_)
		similarity.append(average_similarity)
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
def optimizeOCS_dense(x):

	nus =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
	coefs = [0.1,0.01,0.001]
	nu_list=[]
	coef_list=[]
	score_list=[]
	for nu_ in nus:
		for coef_ in coefs:
			config = Config.OCS_config()
			model_search = OneClassSVM(**config,nu=nu_,coef0=coef_)
			model_search = model_search.fit(x)
			score_search = np.mean((model_search.score_samples(x)))
			nu_list.append(nu_)
			coef_list.append(coef_)
			score_list.append(score_search)
	df_models = pd.DataFrame()
	df_models['nu']=nu_list
	df_models['coef']=coef_list
	df_models['score']=score_list
	print(df_models)

	diff=[]
	for i in range(len(df_models)):
		if i==0:
			diff.append(0)
		else:
			diff.append(df_models['score'][i]-df_models['score'][i-1])
	
	df_models['diff']=diff
	nu_best = df_models['nu'][np.argmax(df_models['diff'])]
	coef_best = df_models['coef'][np.argmax(df_models['diff'])]
	return nu_best,coef_best
def optimizeOCS_silhouette(x):

	nus =[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009]
	coefs = [0.1,0.01,0.001]
	nu_list=[]
	coef_list=[]
	score_list=[]
	for nu_ in nus:
		for coef_ in coefs:
			config = Config.OCS_config()
			model_search = OneClassSVM(**config,nu=nu_,coef0=coef_)
			y = model_search.fit_predict(x)
			score_search = silhouette_score(x,y)

			nu_list.append(nu_)
			coef_list.append(coef_)
			score_list.append(score_search)
	df_models = pd.DataFrame()
	df_models['nu']=nu_list
	df_models['coef']=coef_list
	df_models['score']=score_list
	print(df_models)


	nu_best = df_models['nu'][np.argmax(df_models['score'])]
	coef_best = df_models['coef'][np.argmax(df_models['score'])]

	return nu_best,coef_best
class searchCV():
    def gridSearchCV(model_base):
        param_dict = model_base.get_params()
        print(param_dict)
        #param_dict = {'ccp_alpha': [0.0,0.1], 'min_impurity_decrease': [0.0,0.01], 'min_samples_leaf': [1,2],  'min_weight_fraction_leaf': [0.0,0.1]}
        #param_dict = {  'rbm_learning_rate': [0.001,0.005,0.1],'par__epsilon': [0.05,0.1],  'par__tol': [0.001,0.01]}
        param_dict = {  'nu': [0.2,0.4,0.6,0.8],'tol': [1e-3,1e-4,5e-5,1e-5]}
        #print(param_dict)
        model = GridSearchCV(model_base,param_dict,score='f1')
        return model
    def halvingGridSearchCV(model_base):
        #param_dict = model_base.get_params()
        #param_dict = {  'rbm__learning_rate': [0.001,0.005,0.1],'par__epsilon': [0.05,0.1],  'par__tol': [0.001,0.01]}
        param_dict = {  'nu': [0.2,0.4,0.6,0.8],'tol': [1e-3,1e-4,5e-5,1e-5]}
        #param_dict ={'ccp_alpha': [0.0,0.1], 'min_impurity_decrease': [0.0,0.01], 'min_samples_leaf': [1,2],  'min_weight_fraction_leaf': [0.0,0.1]}
        #print(param_dict)
        model = HalvingGridSearchCV(model_base,param_dict,min_resources=2,score='f1')
        return model