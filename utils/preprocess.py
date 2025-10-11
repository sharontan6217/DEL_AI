from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import models
from models import DBSCAN
import utils
from utils import AEC
from statistics import median,median_low,median_high,geometric_mean,harmonic_mean,quantiles
import gc

scaler =StandardScaler()
class preprocess():
    def descriptors(df_orig,samples):

        df_orig=df_orig[(df_orig['S1_SUM']>0)&(df_orig['Richness_SUM']>0)&(df_orig['S1_COUNT']>0)&(df_orig['Richness_COUNT']>0)&(df_orig['S1_STDEV']>0)&(df_orig['Richness_STDEV']>0)]
        df_orig['S1_ind']  = df_orig['S1_SUM']/df_orig['S1_STDEV']
        df_orig['Richness_ind']  = df_orig['Richness_SUM']/df_orig['Richness_STDEV']
        df_orig['S1_Richness_efficiency']= df_orig['Richness_SUM']/df_orig['S1_SUM']
        df_orig['S1_Richness_balance']=abs(df_orig['Richness_ind']/df_orig['S1_ind'].apply(np.log))
        df_orig = df_orig[['CodeA','CodeB','CodeC','Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT','S1_ind','Richness_ind','S1_Richness_balance','S1_Richness_efficiency']]
        
        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples[i][0]
                codeB=samples[i][1]
                codeC=samples[i][2]
                print(df_orig [(df_orig ['CodeC']==codeC) & (df_orig ['CodeB']==codeB) & (df_orig ['CodeA']==codeA)])

        return df_orig
    def dataStandardize(df,samples):

        print(df.columns)
        cols = ['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']
        for col in cols:
            df[col+'_orig']=df[col]
        df_ = df[cols]

        df_normalized  = pd.DataFrame(scaler.fit_transform(df_.values),columns=df_.columns,index=df_.index)
        print(df_normalized.columns)
        df = df.drop(cols,axis=1)
        df_normalized = pd.concat((df,df_normalized),axis=1)


        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples[i][0]
                codeB=samples[i][1]
                codeC=samples[i][2]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])


        return df_normalized
    def dataPreprocess_filter(df,samples,opt):
        df=df[(df['S1_SUM']>0)&(df['Richness_SUM']>0)&(df['S1_COUNT']>0)&(df['Richness_COUNT']>0)]

        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples[i][0]
                codeB=samples[i][1]
                codeC=samples[i][2]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])
        scope_count = int(len(df)*0.1)
        df=df.nlargest(scope_count , ['S1_COUNT','Richness_COUNT'])

        print('-------------------------s1 and Richness count----------------------')
        print(len(df))

        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples[i][0]
                codeB=samples[i][1]
                codeC=samples[i][2]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])

        scope_balance = int(len(df)*0.4)

        df=df.nlargest(scope_balance , ['S1_ind','Richness_ind'])
        print('-------------------------s1 and Richness balance----------------------')
        print(len(df))

        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples[i][0]
                codeB=samples[i][1]
                codeC=samples[i][2]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])

        scope_balance_ind = int(len(df)*0.9)
        if opt.amplify_deviation_filtering.lower()=='yes':
            scope_balance_ind = int(len(df)*0.8)
        df=df.nlargest(scope_balance_ind , 'S1_Richness_balance')
        print('-------------------------s1 and Richness balance----------------------')
        print(len(df))

        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples[i][0]
                codeB=samples[i][1]
                codeC=samples[i][2]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])
        scope_total_Richness = int(len(df)*0.4)

        df=df.nlargest(scope_total_Richness , ['Richness_SUM','S1_SUM'])
        print('-------------------------total Richness----------------------')
        print(len(df))

        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples[i][0]
                codeB=samples[i][1]
                codeC=samples[i][2]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])
    

        df=df[(df['S1_ind']>1)&(df['Richness_ind']>1)]

        print(len(df))
        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples[i][0]
                codeB=samples[i][1]
                codeC=samples[i][2]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])



        return df
    def dataPreprocess_rank(df,samples,opt):

        scope_balance = int(len(df)*0.8)

        df=df.nlargest(scope_balance , ['S1_ind','Richness_ind'])
        print('-------------------------s1 and Richness balance----------------------')
        print(len(df))
        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples[i][0]
                codeB=samples[i][1]
                codeC=samples[i][2]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])

        scope_balance_ind = int(len(df)*0.8)

        df=df.nlargest(scope_balance_ind , 'S1_Richness_balance')
        print('-------------------------s1 and Richness balance----------------------')
        print(len(df))

        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples[i][0]
                codeB=samples[i][1]
                codeC=samples[i][2]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])

        scope_count = int(len(df)*0.5)
        
        df=df.nlargest(scope_count , ['S1_COUNT','Richness_COUNT'])

        print('-------------------------s1 and Richness count----------------------')
        print(len(df))

        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples[i][0]
                codeB=samples[i][1]
                codeC=samples[i][2]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])

        
        scope_total_Richness = int(len(df)*0.5)

        df=df.nlargest(scope_total_Richness , ['Richness_SUM','S1_SUM'])
        print('-------------------------total Richness----------------------')
        print(len(df))
        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples[i][0]
                codeB=samples[i][1]
                codeC=samples[i][2]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])


        quasi_richness_stdev = [q for q in quantiles((df['Richness_STDEV']),n=20)][-1]
        quasi_s1_stdev = [q for q in quantiles((df['S1_STDEV']),n=20)][-1]

        df=df[(df['Richness_STDEV']<quasi_richness_stdev )|(df['S1_STDEV']<quasi_s1_stdev )]
        
        if opt.amplify_deviation_filtering.lower()=='yes':
            quasi_richness_stdev = [q for q in quantiles((df['Richness_STDEV']),n=20)][-1]
            quasi_s1_stdev = [q for q in quantiles((df['S1_STDEV']),n=20)][-1]

            df=df[(df['Richness_STDEV']<quasi_richness_stdev )|(df['S1_STDEV']<quasi_s1_stdev )]


        df=df[(df['S1_ind']>1)&(df['Richness_ind']>1)]

        print(len(df))
        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples[i][0]
                codeB=samples[i][1]
                codeC=samples[i][2]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])



        return df

    def outlierFiltering(x,df,round,samples):        
        gc.collect()
        
        n = round
        
        eps_aec,min_samples_aec = AEC(x)
        model_classification = DBSCAN(eps=eps_aec,min_samples=min_samples_aec)

        y_predict = model_classification.fit_predict(x)
        df['class'] = y_predict
        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples[i][0]
                codeB=samples[i][1]
                codeC=samples[i][2]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])
        df_filtered=df[df['class']!=-1]
        print(df[df['class']==-1])
        print(len(df_filtered))
        '''
        with open (temp_dir+'eps_minSamples_'+str(n)+'.log','w') as f:
            f.write(str(eps_)+','+str(min_samples_))
            f.close()
        '''
        return df_filtered