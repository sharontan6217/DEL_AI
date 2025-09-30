
import numpy as np

from scipy.spatial.distance import euclidean
class utils():

    def logProbability(model,x):
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
