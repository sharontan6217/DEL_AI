
import numpy as np
import pandas as pd
import random
import sklearn
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,ShuffleSplit,HalvingGridSearchCV
import models
from models import model_base,oneclassSVM
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, KernelDensity
from scipy.spatial.distance import sqeuclidean,jaccard,canberra,cdist,euclidean
class utils():
    def AEC(x):
        kde=KernelDensity(kernel='gaussian').fit(x)
        dens=kde.score_samples(x)
        #print(dens)            

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
            #print(len(dist))
        print(len(dist))
        average_dist=max(dist)
        print('average distance is: ', average_dist)
        eps_aec=average_dist/(2*min_samples_aec)

        print("eps is: ", eps_aec)
            
        
        print("min_samples are: ", min_samples_aec)
        return eps_aec,min_samples_aec
    def logProbability(model,x):

        score = model.score_samples(x)
        score[score==0]=1
        f = np.log(score)

        return f
    
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


class searchCV():
    def gridSearchCV(model_base):
        param_dict = model_base.get_params()
        print(param_dict)
        #param_dict = {'ccp_alpha': [0.0,0.1], 'min_impurity_decrease': [0.0,0.01], 'min_samples_leaf': [1,2],  'min_weight_fraction_leaf': [0.0,0.1]}
        #param_dict = {  'rbm_learning_rate': [0.001,0.005,0.1],'par__epsilon': [0.05,0.1],  'par__tol': [0.001,0.01]}
        param_dict = {  'nu': [0.2,0.4,0.6,0.8],'tol': [1e-3,1e-4,5e-5,1e-5]}
        #print(param_dict)
        model = GridSearchCV(model_base,param_dict)
        return model
    def halvingGridSearchCV(model_base):
        #param_dict = model_base.get_params()
        #param_dict = {  'rbm__learning_rate': [0.001,0.005,0.1],'par__epsilon': [0.05,0.1],  'par__tol': [0.001,0.01]}
        param_dict = {  'nu': [0.2,0.4,0.6,0.8],'tol': [1e-3,1e-4,5e-5,1e-5]}
        #param_dict ={'ccp_alpha': [0.0,0.1], 'min_impurity_decrease': [0.0,0.01], 'min_samples_leaf': [1,2],  'min_weight_fraction_leaf': [0.0,0.1]}
        #print(param_dict)
        model = HalvingGridSearchCV(model_base,param_dict,min_resources=2)
        return model