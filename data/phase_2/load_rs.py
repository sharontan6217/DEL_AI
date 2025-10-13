import pandas as pd
import os
df_total = pd.DataFrame()
data_dir = 'C:/Users/sharo/Documents/chemistry/data/phase_2/'
gc.collect()
for r,d,f in os.walk(data_dir):
    print(r)
    for i in range(len(f)):
        f_ = f[i]
        
        if f_.split(".")[1]=="rs":
            print(f_)
            chunks=[]
            for chunk in pd.read_csv(r+f_,chunksize=10000,low_memory=False ):
                chunks.append(chunk)
            df = pd.concat(chunks,axis=0,ignore_index=True)
            df = df.sort_values(by=['CodeA','CodeB','CodeC'])
            print(df_total.columns)
            if len(df_total) == 0:
                df_total = df
                df_total['S1_SUM'] = 0
                df_total[str(f_)]=df['S1']
                df_total = df_total.drop("S1",axis=1)
                #print(df_total['S1_SUM'])
            else:
                df_total = df_total.sort_values(by=['CodeA','CodeB','CodeC'])
                df_total = df_total.merge(df,how='left',on=['CodeA','CodeB','CodeC'],suffixes=('', '_'+str(f_)))
                #print(df_total['S1_SUM'])

print(len(df_total))
df_merge=df_total

for r,d,f in os.walk(data_dir):
    print(r)
    for i in range(len(f)):
        f_ = f[i]
        if f_.split(".")[1]=="erh":
            print(f_)
            df = pd.read_csv(r+f_,names=['CodeA','CodeB','CodeC','S1','Richness'] )
            df = df.sort_values(by=['CodeA','CodeB','CodeC'])
            df = df.drop("S1",axis=1)
            df_merge = df_merge.merge(df,how='left',on=['CodeA','CodeB','CodeC'],suffixes=('', '_'+str(f_)))
            print(type(df_merge))
print(len(df_merge))
df_merge.fillna(0, inplace=True)
df_merge['RICHNESS_SUM']=0
for col in df_merge.columns:
    if ".rs" in col:
        df_merge['S1_SUM'] = df_merge['S1_SUM']+df_merge[str(col)]

    elif '.erh' in col:
         df_merge['RICHNESS_SUM']=df_merge['RICHNESS_SUM']+df_merge[str(col)]
#print(df_merge['RICHNESS_SUM'])
print(df_merge[(df_merge["CodeC"]==330) & (df_merge["CodeB"]==33) & (df_merge["CodeA"]==2)])
df_merge = df_merge.sort_values(by=['S1_SUM','RICHNESS_SUM'],ascending=False)
df_merge['RICHNESS_RANK'] =  df_merge['RICHNESS_SUM'].rank(ascending=False)
df_merge['S1_RANK'] =  df_merge['S1_SUM'].rank(ascending=False)

df_merge = df_merge.reset_index()
print(len(df_merge))
df_merge.to_csv(data_dir+"total.csv",chunksize=10000)