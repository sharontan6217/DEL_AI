import pandas as pd
import numpy as np
import utils
from utils import utils
import gc
def similarity(df_erh_insr_ipca,opt):
    gc.collect()
    model_names = ['kmeans','AgglmerativeClustering']


    print(df_erh_insr_ipca.columns)

    df = df_erh_insr_ipca
    print(df.columns)

    y_min_1355 = min(df['total_pY1355_ipca'] )
    y_min_1361 = min(df['total_pY1361_ipca'] )
    x_min_1355 = df['total_insr_1355_ipca']  [np.argmin(df['total_pY1355_ipca'])]
    x_min_1361 = df['total_insr_1361_ipca']  [np.argmin(df['total_pY1361_ipca'])]   
    x_min = min(df['total_insr'])
    print(x_min_1355,y_min_1355)
    print(x_min_1361,y_min_1361 )
    erh_array_1355 = np.array(df[['total_insr_1355_ipca','total_pY1355_ipca']])
    erh_array_1361 = np.array(df[['total_insr_1361_ipca','total_pY1361_ipca']])
    c_1355 = [x_min_1355,y_min_1355]
    c_1361 = [x_min_1361,y_min_1361]
    x_min = min(df['total_insr'])
    df['Similarity_point_pY1355'] = utils.distanceCP(erh_array_1355,c_1355)
    df['Similarity_point_pY1361'] = utils.distanceCP(erh_array_1361,c_1361)
    df['Similarity_centralLine_pY1355'] = ( df['total_insr_1355_ipca']-x_min_1355)
    df['Similarity_centralLine_pY1361'] = ( df['total_insr_1361_ipca']-x_min_1361)

    df['Similarity_pY1355']=df['Similarity_point_pY1355'] + abs(df['Similarity_centralLine_pY1355'])
    df['Similarity_pY1361']=df['Similarity_point_pY1361'] + abs(df['Similarity_centralLine_pY1361'])





    x_min_1355_mixed = min(df['total_insr_pY1355_mixed_ipca'])
    x_min_1361_mixed= min(df['total_insr_pY1361_mixed_ipca'])
    y_min_1355_mixed = min(df['total_pY1355_mixed_ipca'])
    y_min_1361_mixed= min(df['total_pY1361_mixed_ipca'])
    print(x_min_1355_mixed,x_min_1361_mixed,y_min_1355_mixed,y_min_1361_mixed)
    erh_array_1355_mixed = np.array(df[['total_insr_pY1355_mixed_ipca','total_pY1355_mixed_ipca']])
    erh_array_1361_mixed = np.array(df[['total_insr_pY1361_mixed_ipca','total_pY1361_mixed_ipca']])
    c_1355_mixed = [x_min_1355_mixed,y_min_1355_mixed]
    c_1361_mixed = [x_min_1361_mixed,y_min_1361_mixed]
    df['Similarity_point_pY1355_mixed'] = utils.distanceCP(erh_array_1355_mixed,c_1355_mixed)
    df['Similarity_point_pY1361_mixed'] = utils.distanceCP(erh_array_1361_mixed,c_1361_mixed)

    if x_min_1355_mixed<y_min_1355_mixed:
        df['Similarity_antiinsr_1355'] = 1/(df['total_insr_pY1355_mixed_ipca']-x_min_1355_mixed)
        x_min_1355_mixed_ipca = df['total_insr_pY1355_mixed_ipca']  [np.argmin(df['total_pY1355_mixed_ipca'])]
        df['Similarity_centralLine_pY1355_mixed_ipca'] = ( df['total_insr_pY1355_mixed_ipca']-x_min_1355_mixed_ipca)

    else:
        df['Similarity_antiinsr_1355'] = 1/(df['total_pY1355_mixed_ipca']-y_min_1355_mixed)
        y_min_1355_mixed_ipca = df['total_pY1355_mixed_ipca']  [np.argmin(df['total_insr_pY1355_mixed_ipca'])]
        df['Similarity_centralLine_pY1355_mixed_ipca'] = ( df['total_pY1355_mixed_ipca']-y_min_1355_mixed_ipca)
    if x_min_1361_mixed<y_min_1361_mixed:
        df['Similarity_antiinsr_1361'] = 1/(df['total_insr_pY1361_mixed_ipca']-x_min_1361_mixed)
        x_min_1361_mixed_ipca = df['total_insr_pY1361_mixed_ipca']  [np.argmin(df['total_pY1361_mixed_ipca'])]  
        df['Similarity_centralLine_pY1361_mixed_ipca'] = ( df['total_insr_pY1361_mixed_ipca']-x_min_1361_mixed_ipca)
    else:
        df['Similarity_antiinsr_1361'] = 1/(df['total_pY1361_mixed_ipca']-y_min_1355_mixed)
        y_min_1361_mixed_ipca = df['total_pY1361_mixed_ipca']  [np.argmin(df['total_insr_pY1361_mixed_ipca'])]
        df['Similarity_centralLine_pY1361_mixed_ipca'] = ( df['total_pY1361_mixed_ipca']-y_min_1361_mixed_ipca)
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
    df_similarity=pd.concat((df_0,df_1),axis=0)	

    if opt.level.lower().replace(' ','')=='level1':


        df_similarity = df_similarity[(df_similarity['Similarity_centralLine_pY1355'] >=0)|(df_similarity['Similarity_centralLine_pY1361']>=0)]


        if x_min_1355_mixed<y_min_1355_mixed:
            df_similarity = df_similarity[(df_similarity['Similarity_centralLine_pY1355_mixed_ipca']<=0)]
            if x_min_1361_mixed<y_min_1361_mixed:
                df_similarity = df_similarity[(df_similarity['Similarity_centralLine_pY1361_mixed_ipca']<=0)]
            else:
                df_similarity = df_similarity[(df_similarity['Similarity_centralLine_pY1361_mixed_ipca']>=0)]
        else:
            df_similarity = df_similarity[(df_similarity['Similarity_centralLine_pY1355_mixed_ipca']>=0)]
            if x_min_1361_mixed<y_min_1361_mixed:
                df_similarity = df_similarity[(df_similarity['Similarity_centralLine_pY1361_mixed_ipca']<=0)]
            else:
                df_similarity = df_similarity[(df_similarity['Similarity_centralLine_pY1361_mixed_ipca']>=0)]

    return df_similarity