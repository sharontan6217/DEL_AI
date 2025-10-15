import pandas as pd
import numpy as np
import utils
from utils import utils
import models
from models import ipca
import gc
def similarity(df_erh_insr_ipca,opt):
    gc.collect()
    model_names = ['kmeans','AgglmerativeClustering']
    df = df_erh_insr_ipca
    '''
    df['total_pY1355_mixed_ipca']= df['total_pY1355']
    total_pY1361 = df['total_pY1361']
    total_insr = df['total_insr']
    x_min = min(total_insr)
    y_min_1355 = min(total_insr)
    y_min_1361 = min(total_insr)
    print(x_min,y_min_1355,min(total_pY1355))
    print(x_min,y_min_1361,min(total_pY1361) )
    erh_array_1355 = np.array(df[['total_insr','total_pY1355']])
    erh_array_1361 = np.array(df[['total_insr','total_pY1361']])
    c_1355 = [x_min,y_min_1355]
    c_1361 = [x_min,y_min_1361]
    #df['Similarity_pY1355_point'] = utils.distanceCP(erh_array_1355,c_1355)
    #df['Similarity_pY1361_point'] = utils.distanceCP(erh_array_1361,c_1361)
    df['Similarity_centralLine'] = 1/abs(total_insr-x_min)
    #df['Similarity_pY1355'] = df['Similarity_pY1355_point']+ df['Similarity_centralLine'] 
    #df['Similarity_pY1361'] = df['Similarity_pY1361_point']+ df['Similarity_centralLine'] 

    '''
    x_normalized_1355 = np.array(df[['total_insr_normalized','total_pY1355_normalized']])    
 
    #x_normalized_1355 = np.array(df[['total_insr','total_pY1355']])    
    #x_normalized_1361 = np.array(df[['total_insr','total_pY1361']])   
    x_ipca_1355 = ipca.ipca(x_normalized_1355,2 )

    df['total_insr_1355_ipca'] = x_ipca_1355[:,0] 
    df['total_pY1355_ipca'] = x_ipca_1355[:,1] 
    x_normalized_1361 = np.array(df[['total_insr_normalized','total_pY1361_normalized']])  
    x_ipca_1361 = ipca.ipca(x_normalized_1361,2 )
    df['total_insr_1361_ipca'] = x_ipca_1361[:,0] 
    df['total_pY1361_ipca'] = x_ipca_1361[:,1] 
    erh_array_1355 = np.array(df[['total_insr_1355_ipca','total_pY1355_ipca']])
    erh_array_1361 = np.array(df[['total_insr_1361_ipca','total_pY1361_ipca']])

    y_min_1355 = min(df['total_pY1355_ipca'] )
    y_min_1361 = min(df['total_pY1361_ipca'] )
    x_min_1355 = df['total_insr_1355_ipca']  [np.argmin(df['total_pY1355_ipca'])]
    x_min_1361 = df['total_insr_1361_ipca']  [np.argmin(df['total_pY1361_ipca'])]   
    x_min = min(df['total_insr'])
    print(x_min_1355,y_min_1355)
    print(x_min_1361,y_min_1361 )
    c_1355 = [x_min_1355,y_min_1355]
    c_1361 = [x_min_1361,y_min_1361]
    df['Similarity_point_pY1355'] = utils.distanceCP(erh_array_1355,c_1355)
    df['Similarity_point_pY1361'] = utils.distanceCP(erh_array_1361,c_1361)
    df['Similarity_centralLine_pY1355'] = ( df['total_insr_1355_ipca']-x_min_1355)
    df['Similarity_centralLine_pY1361'] = ( df['total_insr_1361_ipca']-x_min_1361)

    df['Similarity_pY1355']=df['Similarity_point_pY1355'] + abs(df['Similarity_centralLine_pY1355'])
    df['Similarity_pY1361']=df['Similarity_point_pY1361'] + abs(df['Similarity_centralLine_pY1361'])
    #df['Similarity_antiinsr'] = abs(df['total_insr']-x_min)



    x_normalized_1355_mixed = np.array(df[['total_insr_pY1355_mixed_normalized','total_pY1355_mixed_normalized']]) 

    #x_normalized_1355_mixed = np.array(df[['total_insr','total_pY1355']])    
    #x_normalized_1361_mixed = np.array(df[['total_insr','total_pY1361']])   
    x_ipca_1355_mixed= ipca.ipca(x_normalized_1355_mixed,2 )
    df['total_insr_pY1355_mixed_ipca'] = x_ipca_1355_mixed[:,0] 
    df['total_pY1355_mixed_ipca'] = x_ipca_1355_mixed[:,1] 
    x_normalized_1361_mixed = np.array(df[['total_insr_pY1361_mixed_normalized','total_pY1361_mixed_normalized']])  
    x_ipca_1361_mixed = ipca.ipca(x_normalized_1361_mixed,2 )
    df['total_insr_pY1361_mixed_ipca'] = x_ipca_1361_mixed[:,0] 
    df['total_pY1361_mixed_ipca'] = x_ipca_1361_mixed[:,1]
    x_min_1355_mixed = min(df['total_insr_pY1355_mixed_ipca'])
    x_min_1361_mixed = min(df['total_insr_pY1361_mixed_ipca'])
    df['Similarity_antiinsr_1355'] = 1/(df['total_insr_pY1355_mixed_ipca']-x_min_1355_mixed)
    df['Similarity_antiinsr_1361'] = 1/(df['total_insr_pY1361_mixed_ipca']-x_min_1361_mixed)
    y_min_1355_mixed_ipca = min(df['total_pY1355_mixed_ipca'] )
    y_min_1361_mixed_ipca  = min(df['total_pY1361_mixed_ipca'] )
    x_min_1355_mixed_ipca  = df['total_insr_pY1355_mixed_ipca']  [np.argmin(df['total_pY1355_mixed_ipca'])]
    x_min_1361_mixed_ipca  = df['total_insr_pY1361_mixed_ipca']  [np.argmin(df['total_pY1361_mixed_ipca'])]   
    print(x_min_1355_mixed_ipca,x_min_1361_mixed_ipca)

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
		
    df['Similarity_centralLine_pY1355_mixed'] = ( df['total_insr_pY1355_mixed_ipca']-x_min_1355_mixed_ipca)
    df['Similarity_centralLine_pY1361_mixed'] = ( df['total_insr_pY1361_mixed_ipca']-x_min_1361_mixed_ipca)
    if opt.level.lower().replace(' ','')=='level1':
        df = df[(df['Similarity_centralLine_pY1355']>=0)|(df['Similarity_centralLine_pY1361']>=0) ]
        df = df[(df['Similarity_centralLine_pY1355_mixed']>=0)&(df['Similarity_centralLine_pY1361_mixed']>=0) ]
    '''

        


        

        
    y_min_1355_mixed = df['total_pY1355_mixed_ipca'] [np.argmin(df['total_insr_pY1355_mixed_ipca'])]
    y_min_1361_mixed = df['total_pY1361_mixed_ipca'] [np.argmin(df['total_insr_pY1361_mixed_ipca'])]   
    df['Similarity_centralLine_pY1355_mixed'] = ( df['total_pY1355_mixed_ipca']-y_min_1355_mixed)
    df['Similarity_centralLine_pY1361_mixed'] = ( df['total_pY1361_mixed_ipca']-y_min_1361_mixed)
    y_min_1355_mixed_ipca = min(df['total_pY1355_mixed_ipca'] )
    y_min_1361_mixed_ipca = min(df['total_pY1361_mixed_ipca'] )
    x_min_1355_mixed_ipca = df['total_insr_pY1355_mixed_ipca'][np.argmin(df['total_pY1355_mixed_ipca'])]
    x_min_1361_mixed_ipca = df['total_insr_pY1361_mixed_ipca'][np.argmin(df['total_pY1361_mixed_ipca'])]    
    df['Similarity_centralLine_pY1355_mixed'] = ( df['total_insr_pY1355_mixed_ipca']-x_min_1355_mixed_ipca)
    df['Similarity_centralLine_pY1361_mixed'] = ( df['total_insr_pY1361_mixed_ipca']-x_min_1361_mixed_ipca)
    df = df[(df['Similarity_centralLine_pY1355_mixed']>=0)|(df['Similarity_centralLine_pY1361_mixed']>=0) ]
    print(x_min_1355,y_min_1355)
    print(x_min_1361,y_min_1361 )
    print(x_min_1361_orig,x_min_1355_orig )
    erh_array_1355 = np.array(df[['total_insr_pY1355_mixed_ipca','total_pY1355_mixed_ipca']])
    erh_array_1361 = np.array(df[['total_insr_pY1361_mixed_ipca','total_pY1361_mixed_ipca']])

    c_1355 = [x_min_1355,y_min_1355]
    c_1361 = [x_min_1361,y_min_1361]
    #x_min = min(df['total_insr'])
    df['Similarity_point_pY1355'] = utils.distanceCP(erh_array_1355,c_1355)
    df['Similarity_point_pY1361'] = utils.distanceCP(erh_array_1361,c_1361)
    df['Similarity_centralLine_pY1355'] = ( df['total_insr_pY1355_mixed_ipca']-x_min_1355_ipca)
    df['Similarity_centralLine_pY1361'] = ( df['total_insr_pY1361_mixed_ipca']-x_min_1361_ipca)
    df['Similarity_pY1355']=df['Similarity_point_pY1355'] + abs(df['Similarity_centralLine_pY1355'])
    df['Similarity_pY1361']=df['Similarity_point_pY1361'] + abs(df['Similarity_centralLine_pY1361'])
    



    median_antiinsr_1355 = np.median(df['Similarity_antiinsr_1355'])
    median_antiinsr_1361 = np.median(df['Similarity_antiinsr_1361'])
    #df = df[df['Similarity_antiinsr_1355']<median_antiinsr_1355 ]
    #df = df[df['Similarity_antiinsr_1361']<median_antiinsr_1361 ]
    print(median_antiinsr_1355,median_antiinsr_1361) 
    median_centralLine_pY1355 = (df['Similarity_centralLine_pY1355'])/2
    median_centralLine_pY1361 = (df['Similarity_centralLine_pY1361'])/2
    print(median_centralLine_pY1355,median_centralLine_pY1361) 

        

        
        
    if x_min_1355<=0:
        df = df[df['Similarity_centralLine_pY1355']<=median_centralLine_pY1355_max ]
    else:
        df = df[df['Similarity_centralLine_pY1355']>median_centralLine_pY1355_min ]     
 
    if x_min_1361<=0:       
        df = df[df['Similarity_centralLine_pY1361']<=median_centralLine_pY1361_max ]
    else:       
        df = df[df['Similarity_centralLine_pY1361']>median_centralLine_pY1361_min ]
    '''
        
    #quasi_similarity_point_pY1355 = [q for q in quantiles((df['Similarity_point_pY1355']),n=5)][0]
    #quasi_similarity_point_pY1361 = [q for q in quantiles((df['Similarity_point_pY1361']),n=5)][0]
    #quasi_similarity_point_all = [q for q in quantiles((df['Similarity_point_all']),n=5)][0]
    #quasi_similarity_centralLine_all = [q for q in quantiles((df['Similarity_centralLine_all']),n=5)][-1]
    #quasi_similarity_variance_ipca = [q for q in quantiles((df['Similarity_variance_ipca']),n=4)][-1]
    #quasi_similarity_centralLine_pY1355 = [q for q in quantiles((df['Similarity_centralLine_pY1355']),n=5)][-1]
    #quasi_similarity_centralLine_pY1361 = [q for q in quantiles((df['Similarity_centralLine_pY1361']),n=5)][-1]
    #print(quasi_similarity_variance_ipca,quasi_similarity_point_all,quasi_similarity_centralLine_all, quasi_similarity_point_pY1355 ,quasi_similarity_point_pY1361,quasi_similarity_centralLine_pY1355 ,quasi_similarity_centralLine_pY1361  )
    #rint(quasi_similarity_variance_ipca,quasi_similarity_point_all,quasi_similarity_centralLine_all)
    #df=df[(df['Similarity_variance_ipca']<quasi_similarity_variance_ipca )]
    #df=df[(df['Similarity_variance_ipca']<quasi_similarity_variance_ipca )|(df['Similarity_point_all']>quasi_similarity_point_all )|(df['Similarity_centralLine_all']<quasi_similarity_centralLine_all )]
    #df=df[(df['Similarity_centralLine_pY1361']<quasi_similarity_centralLine_pY1361 )|(df['Similarity_centralLine_pY1355']<quasi_similarity_centralLine_pY1355 )|(df['Similarity_point_pY1361']>quasi_similarity_point_pY1361 )|(df['Similarity_point_pY1355']>quasi_similarity_point_pY1355 )]
    df_0 = df[df['class']==0]
    df_1 = df[df['class']==1]
    x_0= df_0[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']]
    x_0=np.array(x_0)
    x_1= df_1[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']]
    x_1=np.array(x_1)
    if opt.model_name=='OneClassSVM':
        df_0['Similarity']=df_0['Score_OSVM']
        df_1['Similarity']=df_1['Score_OSVM']        
    else:
        df_0['Similarity']=utils.boundary(x_0,x_1)
        df_1['Similarity']=utils.boundary(x_1,x_0)
    #print(len(df_))
    df_similarity=pd.concat((df_0,df_1),axis=0)
    return df_similarity