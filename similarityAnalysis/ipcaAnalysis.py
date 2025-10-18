import models
from models import ipca
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler, Normalizer, normalize,minmax_scale
import gc
scaler =StandardScaler()
def ipcaAnalysis(df_erh_insr,df_result):
    gc.collect()
    print(df_erh_insr.columns)

    df_erh_insr=df_result.merge(df_erh_insr,how='left',on=['CodeA','CodeB','CodeC'])
    df_erh_insr_ = df_erh_insr[['total_pY1355','total_pY1361','total_insr','total_pY1355_mixed','total_pY1361_mixed','total_insr_pY1355_mixed','total_insr_pY1361_mixed','s1_pY1355','s1_pY1361','s1_insr','s1_pY1355_mixed','s1_pY1361_mixed','s1_insr_pY1355_mixed','s1_insr_pY1361_mixed']]
    df_erh_insr_normalized =  pd.DataFrame(scaler.fit_transform(df_erh_insr_.values),columns=df_erh_insr_.columns+'_normalized',index=df_erh_insr_.index)

    df_erh_insr_ipca = pd.concat((df_erh_insr,df_erh_insr_normalized),axis=1)
    x_normalized_1355 = np.array(df_erh_insr_ipca[['total_insr_normalized','total_pY1355_normalized']])    
 
    #x_normalized_1355 = np.array(df_erh_insr_ipca[['total_insr','total_pY1355']])    
    #x_normalized_1361 = np.array(df_erh_insr_ipca[['total_insr','total_pY1361']])   
    x_ipca_1355 = ipca.ipca(x_normalized_1355,2 )
    print(x_ipca_1355)
    df_erh_insr_ipca['total_insr_1355_ipca'] = x_ipca_1355[:,0] 
    df_erh_insr_ipca['total_pY1355_ipca'] = x_ipca_1355[:,1] 
    x_normalized_1361 = np.array(df_erh_insr_ipca[['total_insr_normalized','total_pY1361_normalized']])  
    x_ipca_1361 = ipca.ipca(x_normalized_1361,2 )
    df_erh_insr_ipca['total_insr_1361_ipca'] = x_ipca_1361[:,0] 
    df_erh_insr_ipca['total_pY1361_ipca'] = x_ipca_1361[:,1] 
    x_normalized_1355_mixed = np.array(df_erh_insr_ipca[['total_insr_pY1355_mixed_normalized','total_pY1355_mixed_normalized']]) 

    #x_normalized_1355_mixed = np.array(df[['total_insr','total_pY1355']])    
    #x_normalized_1361_mixed = np.array(df[['total_insr','total_pY1361']])   
    x_ipca_1355_mixed= ipca.ipca(x_normalized_1355_mixed,2 )
    df_erh_insr_ipca['total_insr_pY1355_mixed_ipca'] = x_ipca_1355_mixed[:,0] 
    df_erh_insr_ipca['total_pY1355_mixed_ipca'] = x_ipca_1355_mixed[:,1] 
    x_normalized_1361_mixed = np.array(df_erh_insr_ipca[['total_insr_pY1361_mixed_normalized','total_pY1361_mixed_normalized']])  
    x_ipca_1361_mixed = ipca.ipca(x_normalized_1361_mixed,2 )
    df_erh_insr_ipca['total_insr_pY1361_mixed_ipca'] = x_ipca_1361_mixed[:,0] 
    df_erh_insr_ipca['total_pY1361_mixed_ipca'] = x_ipca_1361_mixed[:,1]

    x_erh_1355_insr_normalized = np.array(df_erh_insr_ipca[['total_pY1355_normalized','s1_pY1355_normalized','total_insr_normalized','s1_insr_normalized']])    
 
    #x_normalized_1355 = np.array(df_erh_insr_ipca[['total_insr','total_pY1355']])    
    #x_normalized_1361 = np.array(df_erh_insr_ipca[['total_insr','total_pY1361']])   
    x_erh_1355_ipca = ipca.ipca(x_erh_1355_insr_normalized ,3 )
    print(x_erh_1355_ipca.shape)
    df_erh_insr_ipca['pY1355_ipca_x'] = x_erh_1355_ipca[:,0] 
    df_erh_insr_ipca['pY1355_ipca_y'] =  x_erh_1355_ipca[:,1] 
    df_erh_insr_ipca['pY1355_ipca_z'] =  x_erh_1355_ipca[:,2] 

    x_erh_1361_insr_normalized = np.array(df_erh_insr_ipca[['total_pY1361_normalized','s1_pY1361_normalized','total_insr_normalized','s1_insr_normalized']])    
 
    #x_normalized_1355 = np.array(df_erh_insr_ipca[['total_insr','total_pY1355']])    
    #x_normalized_1361 = np.array(df_erh_insr_ipca[['total_insr','total_pY1361']])   
    x_erh_1361_ipca = ipca.ipca(x_erh_1361_insr_normalized ,3 )
    print(x_erh_1361_ipca.shape)
    df_erh_insr_ipca['pY1361_ipca_x'] = x_erh_1361_ipca[:,0] 
    df_erh_insr_ipca['pY1361_ipca_y'] =  x_erh_1361_ipca[:,1] 
    df_erh_insr_ipca['pY1361_ipca_z'] =  x_erh_1361_ipca[:,2] 

    x_erh_insr_1355_normalized_mixed = np.array(df_erh_insr_ipca[['total_pY1355_mixed_normalized','s1_pY1355_mixed_normalized','total_insr_pY1355_mixed_normalized','s1_insr_pY1355_mixed_normalized']]) 

    #x_normalized_1355_mixed = np.array(df[['total_insr','total_pY1355']])    
    #x_normalized_1361_mixed = np.array(df[['total_insr','total_pY1361']])   
    x_erh_1355_ipca_mixed= ipca.ipca(x_erh_insr_1355_normalized_mixed,3 )
    df_erh_insr_ipca['pY1355_mixed_ipca_x'] = x_erh_1355_ipca_mixed[:,0] 
    df_erh_insr_ipca['pY1355_mixed_ipca_y'] = x_erh_1355_ipca_mixed[:,1] 
    df_erh_insr_ipca['pY1355_mixed_ipca_z'] = x_erh_1355_ipca_mixed[:,2] 
    x_erh_insr_1361_normalized_mixed = np.array(df_erh_insr_ipca[['total_pY1361_mixed_normalized','s1_pY1361_mixed_normalized','total_insr_pY1361_mixed_normalized','s1_insr_pY1361_mixed_normalized']]) 

    #x_normalized_1355_mixed = np.array(df[['total_insr','total_pY1355']])    
    #x_normalized_1361_mixed = np.array(df[['total_insr','total_pY1361']])   
    x_erh_1361_ipca_mixed= ipca.ipca(x_erh_insr_1361_normalized_mixed,3 )
    df_erh_insr_ipca['pY1361_mixed_ipca_x'] = x_erh_1361_ipca_mixed[:,0] 
    df_erh_insr_ipca['pY1361_mixed_ipca_y'] = x_erh_1361_ipca_mixed[:,1] 
    df_erh_insr_ipca['pY1361_mixed_ipca_z'] = x_erh_1361_ipca_mixed[:,2] 
    print('ipca columns:')
    print(df_erh_insr_ipca.columns)

    return df_erh_insr_ipca
def ipcaAnalysis_tpor(df_erh_tpor,df_result):
    gc.collect()
    print(df_erh_tpor.columns)

    df_erh_tpor=df_result.merge(df_erh_tpor,how='left',on=['CodeA','CodeB','CodeC'])
    df_erh_tpor_ = df_erh_tpor[['total_pY626','total_tpor','total_pY626_mixed','total_tpor_pY626_mixed','s1_pY626','s1_tpor','s1_pY626_mixed','s1_tpor_pY626_mixed']]
    df_erh_tpor_normalized =  pd.DataFrame(scaler.fit_transform(df_erh_tpor_.values),columns=df_erh_tpor_.columns+'_normalized',index=df_erh_tpor_.index)

    df_erh_tpor_ipca = pd.concat((df_erh_tpor,df_erh_tpor_normalized),axis=1)
    x_normalized_626 = np.array(df_erh_tpor_ipca[['total_tpor_normalized','total_pY626_normalized']])    
 
    #x_normalized_626 = np.array(df_erh_tpor_ipca[['total_tpor','total_pY626']])    

    x_ipca_626 = ipca.ipca(x_normalized_626,2 )
    print(x_ipca_626)
    df_erh_tpor_ipca['total_tpor_626_ipca'] = x_ipca_626[:,0] 
    df_erh_tpor_ipca['total_pY626_ipca'] = x_ipca_626[:,1] 

    x_normalized_626_mixed = np.array(df_erh_tpor_ipca[['total_tpor_pY626_mixed_normalized','total_pY626_mixed_normalized']]) 

    #x_normalized_626_mixed = np.array(df[['total_tpor','total_pY626']])    

    x_ipca_626_mixed= ipca.ipca(x_normalized_626_mixed,2 )
    df_erh_tpor_ipca['total_tpor_pY626_mixed_ipca'] = x_ipca_626_mixed[:,0] 
    df_erh_tpor_ipca['total_pY626_mixed_ipca'] = x_ipca_626_mixed[:,1] 


    x_erh_626_tpor_normalized = np.array(df_erh_tpor_ipca[['total_pY626_normalized','s1_pY626_normalized','total_tpor_normalized','s1_tpor_normalized']])    
 
    #x_normalized_626 = np.array(df_erh_tpor_ipca[['total_tpor','total_pY626']])    

    x_erh_626_ipca = ipca.ipca(x_erh_626_tpor_normalized ,3 )
    print(x_erh_626_ipca.shape)
    df_erh_tpor_ipca['pY626_ipca_x'] = x_erh_626_ipca[:,0] 
    df_erh_tpor_ipca['pY626_ipca_y'] =  x_erh_626_ipca[:,1] 
    df_erh_tpor_ipca['pY626_ipca_z'] =  x_erh_626_ipca[:,2] 


    #x_normalized_626 = np.array(df_erh_tpor_ipca[['total_tpor','total_pY626']])    


    x_erh_tpor_626_normalized_mixed = np.array(df_erh_tpor_ipca[['total_pY626_mixed_normalized','s1_pY626_mixed_normalized','total_tpor_pY626_mixed_normalized','s1_tpor_pY626_mixed_normalized']]) 

    #x_normalized_626_mixed = np.array(df[['total_tpor','total_pY626']])    

    x_erh_626_ipca_mixed= ipca.ipca(x_erh_tpor_626_normalized_mixed,3 )
    df_erh_tpor_ipca['pY626_mixed_ipca_x'] = x_erh_626_ipca_mixed[:,0] 
    df_erh_tpor_ipca['pY626_mixed_ipca_y'] = x_erh_626_ipca_mixed[:,1] 
    df_erh_tpor_ipca['pY626_mixed_ipca_z'] = x_erh_626_ipca_mixed[:,2] 

    #x_normalized_626_mixed = np.array(df[['total_tpor','total_pY626']])    


    return df_erh_tpor_ipca