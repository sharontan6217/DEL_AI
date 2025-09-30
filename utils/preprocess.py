from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, KernelDensity
import pandas as pd
import numpy as np
import models
from models import DBSCAN
from scipy.spatial.distance import sqeuclidean,jaccard,canberra,cdist,euclidean

import gc
scaler =StandardScaler()
class preprocess():
    def descriptors(df_orig):

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