from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import models
from sklearn.cluster import DBSCAN
from utils import utils
from statistics import median,median_low,median_high,geometric_mean,harmonic_mean,quantiles
import gc

scaler =StandardScaler()
class preprocess():
    def descriptors(df_orig,samples):
        print(df_orig.columns)
        df_orig=df_orig[(df_orig['S1_SUM']>0)&(df_orig['Richness_SUM']>0)&(df_orig['S1_COUNT']>0)&(df_orig['Richness_COUNT']>0)&(df_orig['S1_STDEV']>0)&(df_orig['Richness_STDEV']>0)]
        df_orig['S1_ind']  = df_orig['S1_SUM']/df_orig['S1_STDEV']
        df_orig['Richness_ind']  = df_orig['Richness_SUM']/df_orig['Richness_STDEV']
        df_orig['S1_Richness_efficiency']= df_orig['Richness_SUM']/df_orig['S1_SUM']
        df_orig['S1_Richness_balance']=abs((df_orig['Richness_ind']/df_orig['S1_ind']).apply(np.log))  
        df_orig = df_orig[['CodeA','CodeB','CodeC','Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT','S1_ind','Richness_ind','S1_Richness_balance','S1_Richness_efficiency']]
        print(df_orig)
        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples['CodeA'][i]
                codeB=samples['CodeB'][i]
                codeC=samples['CodeC'][i]
                print(codeA,codeB,codeC)
                print(df_orig [(df_orig ['CodeC']==codeC) & (df_orig ['CodeB']==codeB) & (df_orig ['CodeA']==codeA)])

        return df_orig
    def dataStandardize(df,samples):
        print(df.columns)
        cols = ['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']
        for col in cols:
            df[col+'_orig']=df[col]

        df_ = df[cols]
        df = df.drop(cols,axis=1)


        df_normalized  = pd.DataFrame(scaler.fit_transform(df_.values),columns=df_.columns,index=df_.index)
        print(df_normalized.columns)
        df_normalized = pd.concat((df,df_normalized),axis=1)


        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples['CodeA'][i]
                codeB=samples['CodeB'][i]
                codeC=samples['CodeC'][i]
                print(df_normalized [(df_normalized ['CodeC']==codeC) & (df_normalized ['CodeB']==codeB) & (df_normalized ['CodeA']==codeA)])


        return df_normalized
    def dataPreprocess_filter(df,samples,opt):
        df=df[(df['S1_SUM']>0)&(df['Richness_SUM']>0)&(df['S1_COUNT']>0)&(df['Richness_COUNT']>0)]
        print('----------preprocess filter for active candidates-------------')
        print(len(df))
        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples['CodeA'][i]
                codeB=samples['CodeB'][i]
                codeC=samples['CodeC'][i]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])
        scope_count = int(len(df)*0.1)
        df=df.nlargest(scope_count , ['S1_COUNT','Richness_COUNT'])

        print('-------------------------s1 and Richness count----------------------')
        print(len(df))

        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples['CodeA'][i]
                codeB=samples['CodeB'][i]
                codeC=samples['CodeC'][i]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])

        scope_balance = int(len(df)*0.4)

        df=df.nlargest(scope_balance , ['S1_ind','Richness_ind'])
        print('-------------------------s1 and Richness balance I----------------------')
        print(len(df))

        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples['CodeA'][i]
                codeB=samples['CodeB'][i]
                codeC=samples['CodeC'][i]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])

        scope_balance_ind = int(len(df)*0.8)
        if opt.amplify_deviation_filtering.lower()=='yes':
            scope_balance_ind = int(len(df)*0.9)
        df=df.nsmallest(scope_balance_ind , 'S1_Richness_balance')
        print('-------------------------s1 and Richness balance II----------------------')
        print(len(df))

        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples['CodeA'][i]
                codeB=samples['CodeB'][i]
                codeC=samples['CodeC'][i]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])
        scope_total_Richness = int(len(df)*0.4)

        df=df.nlargest(scope_total_Richness , ['Richness_SUM','S1_SUM'])
        print('-------------------------total Richness----------------------')
        print(len(df))

        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples['CodeA'][i]
                codeB=samples['CodeB'][i]
                codeC=samples['CodeC'][i]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])
    

        df=df[(df['S1_ind']>1)&(df['Richness_ind']>1)]

        print(len(df))
        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples['CodeA'][i]
                codeB=samples['CodeB'][i]
                codeC=samples['CodeC'][i]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])



        return df
    def dataPreprocess_rank(df,samples,opt):
        print('----------preprocess filter for most proactive candidates-------------')
        scope_balance = int(len(df)*0.7)
        df=df.nlargest(scope_balance , ['S1_ind','Richness_ind'])
        print('-------------------------s1 and Richness balance I----------------------')
        print(len(df))
        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples['CodeA'][i]
                codeB=samples['CodeB'][i]
                codeC=samples['CodeC'][i]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])

        if opt.amplify_deviation_filtering.lower()=='no':
            scope_balance_ind = len(df)
            df=df.nsmallest(scope_balance_ind , 'S1_Richness_balance')

        print('-------------------------s1 and Richness balance II----------------------')
        print(len(df))

        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples['CodeA'][i]
                codeB=samples['CodeB'][i]
                codeC=samples['CodeC'][i]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])

        scope_count = int(len(df)*0.5)
        
        df=df.nlargest(scope_count , ['S1_COUNT','Richness_COUNT'])

        print('-------------------------s1 and Richness count----------------------')
        print(len(df))

        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples['CodeA'][i]
                codeB=samples['CodeB'][i]
                codeC=samples['CodeC'][i]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])

        
        scope_total_Richness = int(len(df)*0.5)

        df=df.nlargest(scope_total_Richness , ['Richness_SUM','S1_SUM'])
        print('-------------------------total Richness----------------------')
        print(len(df))
        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples['CodeA'][i]
                codeB=samples['CodeB'][i]
                codeC=samples['CodeC'][i]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])

        if opt.amplify_deviation_filtering.lower()=='yes':
            quasi_richness_sum = [q for q in quantiles((df['Richness_SUM']),n=5)][-1]
            quasi_s1_sum = [q for q in quantiles((df['S1_SUM']),n=5)][-1]

            df=df[(df['Richness_SUM']>quasi_richness_sum )&(df['S1_SUM']>quasi_s1_sum )]


        quasi_richness_stdev = [q for q in quantiles((df['Richness_STDEV']),n=25)][-1]
        quasi_s1_stdev = [q for q in quantiles((df['S1_STDEV']),n=25)][-1]

        df=df[(df['Richness_STDEV']<quasi_richness_stdev )|(df['S1_STDEV']<quasi_s1_stdev )]

        df=df[(df['S1_ind']>1)&(df['Richness_ind']>1)]


        print(len(df))
        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples['CodeA'][i]
                codeB=samples['CodeB'][i]
                codeC=samples['CodeC'][i]
                print(df [(df ['CodeC']==codeC) & (df ['CodeB']==codeB) & (df ['CodeA']==codeA)])


        return df
class outlierFiltering():
    def outlierFiltering(x,df,round,samples):        
        gc.collect()
        
        n = round
        
        eps_aec,min_samples_aec = utils.AEC(x)
        model_classification = DBSCAN(eps=eps_aec,min_samples=min_samples_aec)

        y_predict = model_classification.fit_predict(x)
        df['class'] = y_predict
        if len(samples)>0:
            for i in range(len(samples)):
                codeA=samples['CodeA'][i]
                codeB=samples['CodeB'][i]
                codeC=samples['CodeC'][i]
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