import pandas as pd
import numpy as np
import os

def dataLoading_filter(data_dir,samples):
    global cols_erh,cols_rs
	
    df_orig = pd.read_csv(data_dir)
    print(len(df_orig))

    columns = df_orig.columns
    cols_rs=[]
    cols_erh= []
    for col in columns:
        if '.rs' in col:
            #print(col)
            cols_rs.append(str(col))
        elif '.erh' in col:
            cols_erh.append(str(col))
    #print(cols_rs)
    df_orig ['S1_COUNT']= (df_orig[cols_rs]>0).sum(axis=1)
        
    df_orig ['Richness_COUNT']= (df_orig[cols_erh]>0).sum(axis=1)

    df_orig['S1_STDEV'] = df_orig[cols_rs].std(axis=1)
    df_orig['Richness_STDEV'] = df_orig[cols_erh].std(axis=1)

    if len(samples)>0:
        for i in range(len(samples)):
            codeA=samples['CodeA'][i]
            codeB=samples['CodeB'][i]
            codeC=samples['CodeC'][i]
            print(df_orig [(df_orig ['CodeC']==codeC) & (df_orig  ['CodeB']==codeB) & (df_orig  ['CodeA']==codeA)])
	
    return df_orig

def dataLoading_rank(data_dir,samples):
    global cols_erh,cols_rs
    df_orig = pd.read_csv(data_dir)
    df_orig.fillna(0, inplace=True)
    print(df_orig[df_orig['class']==1])
    
    df_orig = df_orig[df_orig['class']==1]
    print(df_orig)
    
    df_orig=df_orig[(df_orig['performance_ind_1_total']>0)|(df_orig['performance_ind_0_total']>0)]


    df_orig = df_orig[['CodeA','CodeB','CodeC','Richness_SUM_orig','Richness_STDEV_orig','Richness_COUNT_orig','S1_SUM_orig','S1_STDEV_orig','S1_COUNT_orig']]
    df_orig.columns = df_orig.columns.str.replace('_orig','')
    print(df_orig)
    if len(samples)>0:
        for i in range(len(samples)):
            codeA=samples['CodeA'][i]
            codeB=samples['CodeB'][i]
            codeC=samples['CodeC'][i]
            print(df_orig [(df_orig ['CodeC']==codeC) & (df_orig  ['CodeB']==codeB) & (df_orig  ['CodeA']==codeA)])
	
    print(len(df_orig))

    return df_orig
def erhAnalysis(erh_dir,df_filtered,samples):
	print(len(df_filtered))
	df_merge = df_filtered

	array_1 = [[1,3],[4,6],[7,9],[10,12],[13,15],[16,18]]
	array_2 = [[2,3],[5,6],[8,9],[11,12],[14,15],[17,18]]
	col_1355=[]
	col_insr=[]
	col_1361=[]    
	#df_insr_total=pd.DataFrame()
	for r,d,f in os.walk(erh_dir):
		print(r)
		for m in range(len(array_1)):
			num_1355 =array_1[m][0]
			num_1361 =array_2[m][0]
			num_insr =array_1[m][1]
			for f_ in f:
				if num_1355 == int(f_.split("-")[0]):
					#print('pY1355 col file is: ',f_)
					col_1355.append(f_)
				elif num_1361 == int(f_.split("-")[0]):
					#print('pY1361 col file is: ',f_)
					col_1361.append(f_)
				elif num_insr == int(f_.split("-")[0]):
					#print('INSR col file is: ',f_)
					col_insr.append(f_)
		for i in range(len(f)):
			f_ = f[i]
			if f_.split(".")[1]=="erh":
				if "pY1355" in f_.split(".")[0]:
					print('pY1355 file is: ',f_)

					df_1355 = pd.read_csv(r+f_,names=['CodeA','CodeB','CodeC','S1','Richness'] )
					df_1355 = df_1355.sort_values(by=['CodeA','CodeB','CodeC'])
					col_name = str(f_)
					df_1355[col_name] = df_1355['Richness']
					df_1355  = df_1355.drop("Richness",axis=1)
					df_1355[col_name+'_S1'] = df_1355['S1']
					df_1355 = df_1355.drop("S1",axis=1)
					df_merge = df_merge.merge(df_1355,how='left',on=['CodeA','CodeB','CodeC'])
				elif "pY1361" in f_.split(".")[0]:
					print('pY1361 file is: ',f_)
					df_1361 = pd.read_csv(r+f_,names=['CodeA','CodeB','CodeC','S1','Richness'])
					df_1361 = df_1361.sort_values(by=['CodeA','CodeB','CodeC'])
					col_name = str(f_)
					df_1361[col_name] = df_1361['Richness']
					df_1361  = df_1361.drop("Richness",axis=1)
					df_1361[col_name+'_S1'] = df_1361['S1']
					df_1361 = df_1361.drop("S1",axis=1)
					df_merge = df_merge.merge(df_1361,how='left',on=['CodeA','CodeB','CodeC'])
				elif "INSR" in f_.split(".")[0]:
					print('INSR file is: ',f_)
					df_ins = pd.read_csv(r+f_,names=['CodeA','CodeB','CodeC','S1','Richness'])
					df_ins = df_ins.sort_values(by=['CodeA','CodeB','CodeC'])
					print(df_ins [(df_ins ['CodeC']==42) & (df_ins ['CodeB']==703) & (df_ins ['CodeA']==327)])
					col_name = str(f_)
					df_ins[col_name] = df_ins['Richness']
					df_ins = df_ins.drop("Richness",axis=1)
					df_ins[col_name+'_S1'] = df_ins['S1']
					df_ins = df_ins.drop("S1",axis=1)
					#df_insr_total=pd.concat((df_insr_total,df_ins),axis=1)
					df_merge = df_merge.merge(df_ins,how='left',on=['CodeA','CodeB','CodeC'])
					print(df_merge [(df_merge ['CodeC']==42) & (df_merge ['CodeB']==703) & (df_merge ['CodeA']==327)])
	#df_insr_total.to_csv("insr.csv")
	df_merge.fillna(0,inplace=True)
	print(df_merge [(df_merge ['CodeC']==1457) & (df_merge ['CodeB']==210) & (df_merge ['CodeA']==104)])
	df_col = pd.DataFrame()
	df_col['col_1355']=col_1355
	df_col['col_1361']=col_1361
	df_col['col_insr']=col_insr
	print(df_col)
	
	df_merge['performance_ind_0_total']=0
	df_merge['richness_1355_0']=0
	df_merge['richness_1355_1']=0
	df_merge['performance_ind_1_total']=0
	df_merge['richness_1361_0']=0
	df_merge['richness_1361_1']=0
	df_merge['richness_insr_1355_0']=0
	df_merge['richness_insr_1355_1']=0
	df_merge['richness_insr_1361_0']=0
	df_merge['richness_insr_1361_1']=0
	df_merge['s1_1355_0']=0
	df_merge['s1_1355_1']=0
	df_merge['s1_1361_0']=0
	df_merge['s1_1361_1']=0
	df_merge['s1_insr_1355_0']=0
	df_merge['s1_insr_1355_1']=0
	df_merge['s1_insr_1361_0']=0
	df_merge['s1_insr_1361_1']=0

	for i in range(len(df_merge)):
		richiness_1355_1 = 0.00
		richiness_1355_0 = 0.00
		richiness_insr_1355_1 = 0.00
		richiness_insr_1355_0 = 0.00
		richiness_1361_1 = 0.00
		richiness_1361_0 = 0.00
		richiness_insr_1361_1 = 0.00
		richiness_insr_1361_0 = 0.00
		s1_1355_1 = 0.00
		s1_1355_0 = 0.00
		s1_insr_1355_1 = 0.00
		s1_insr_1355_0 = 0.00
		s1_1361_1 = 0.00
		s1_1361_0 = 0.00
		s1_insr_1361_1 = 0.00
		s1_insr_1361_0 = 0.00
		performance_total_1355 = 0
		performance_total_1361 = 0
		for j in range(len(df_col)):
			col_1355_ = df_col['col_1355'].values[j]
			col_1361_ = df_col['col_1361'].values[j]
			col_insr_ = df_col['col_insr'].values[j]
			col_1355_S1 = df_col['col_1355'].values[j]+'_S1'
			col_1361_S1 = df_col['col_1361'].values[j]+'_S1'
			col_insr_S1 = df_col['col_insr'].values[j]+'_S1'
			#print(col_1355_,col_insr_)   
			if df_merge[col_1355_].values[i]>=df_merge[col_insr_].values[i]:
				richiness_1355_1+=df_merge[col_1355_].values[i]
				richiness_insr_1355_1+=df_merge[col_insr_].values[i]
				s1_1355_1+=df_merge[col_1355_S1].values[i]
				s1_insr_1355_1+=df_merge[col_insr_S1].values[i]
				performance_total_1355+=1
			else:
				richiness_1355_0+=df_merge[col_1355_].values[i]
				performance_total_1355+=0
				richiness_insr_1355_0+=df_merge[col_insr_].values[i]
				s1_1355_0+=df_merge[col_1355_S1].values[i]
				s1_insr_1355_0+=df_merge[col_insr_S1].values[i]
			#print(col_1361_,col_insr_)   
			if df_merge[col_1361_].values[i]>=df_merge[col_insr_].values[i]:
				richiness_1361_1+=df_merge[col_1361_].values[i]
				richiness_insr_1361_1+=df_merge[col_insr_].values[i]
				performance_total_1361+=1
				s1_1361_1+=df_merge[col_1361_S1].values[i]
				s1_insr_1361_1+=df_merge[col_insr_S1].values[i]
			else:
				richiness_1361_0+=df_merge[col_1361_].values[i]
				performance_total_1361+=0
				richiness_insr_1361_0+=df_merge[col_insr_].values[i]
				s1_1361_0+=df_merge[col_1361_S1].values[i]
				s1_insr_1361_0+=df_merge[col_insr_S1].values[i]
		df_merge['performance_ind_0_total'].values[i]=performance_total_1355
		df_merge['richness_1355_0'].values[i]=richiness_1355_0
		df_merge['richness_1355_1'].values[i]=richiness_1355_1
		df_merge['performance_ind_1_total'].values[i]=performance_total_1361
		df_merge['richness_1361_0'].values[i]=richiness_1361_0
		df_merge['richness_1361_1'].values[i]=richiness_1361_1
		df_merge['richness_insr_1355_0'].values[i]=richiness_insr_1355_0
		df_merge['richness_insr_1355_1'].values[i]=richiness_insr_1355_1
		df_merge['richness_insr_1361_0'].values[i]=richiness_insr_1361_0
		df_merge['richness_insr_1361_1'].values[i]=richiness_insr_1361_1
		df_merge['s1_1355_0'].values[i]=s1_1355_0
		df_merge['s1_1355_1'].values[i]=s1_1355_1
		df_merge['s1_1361_0'].values[i]=s1_1361_0
		df_merge['s1_1361_1'].values[i]=s1_1361_1
		df_merge['s1_insr_1355_0'].values[i]=s1_insr_1355_0
		df_merge['s1_insr_1355_1'].values[i]=s1_insr_1355_1
		df_merge['s1_insr_1361_0'].values[i]=s1_insr_1361_0
		df_merge['s1_insr_1361_1'].values[i]=s1_insr_1361_1
	print(df_merge [(df_merge ['CodeC']==1457) & (df_merge ['CodeB']==210) & (df_merge ['CodeA']==104)])
	df_merge['total_pY1355_mixed']=0
	df_merge['total_pY1361_mixed']=0
	df_merge['total_insr_pY1355_mixed']=0
	df_merge['total_insr_pY1361_mixed']=0
	df_merge['s1_pY1355_mixed']=0
	df_merge['s1_pY1361_mixed']=0
	df_merge['s1_insr_pY1355_mixed']=0
	df_merge['s1_insr_pY1361_mixed']=0
	for i in range(len(df_merge)):
		if df_merge['performance_ind_0_total'].values[i]>=3:
			df_merge['total_pY1355_mixed'].values[i]=df_merge['richness_1355_1'].values[i]
			df_merge['total_insr_pY1355_mixed'].values[i]=df_merge['richness_insr_1355_1'].values[i]
			df_merge['s1_pY1355_mixed'].values[i]=df_merge['s1_1355_1'].values[i]
			df_merge['s1_insr_pY1355_mixed'].values[i]=df_merge['s1_insr_1355_1'].values[i]
		else:
			df_merge['total_pY1355_mixed'].values[i]=df_merge['richness_1355_0'].values[i]
			df_merge['total_insr_pY1355_mixed'].values[i]=df_merge['richness_insr_1355_0'].values[i]
			df_merge['s1_pY1355_mixed'].values[i]=df_merge['s1_1355_0'].values[i]
			df_merge['s1_insr_pY1355_mixed'].values[i]=df_merge['s1_insr_1355_0'].values[i]
	for i in range(len(df_merge)):
		if df_merge['performance_ind_1_total'].values[i]>=3:
			df_merge['total_pY1361_mixed'].values[i]=df_merge['richness_1361_1'].values[i]
			df_merge['total_insr_pY1361_mixed'].values[i]=df_merge['richness_insr_1361_1'].values[i]
			df_merge['s1_pY1361_mixed'].values[i]=df_merge['s1_1361_1'].values[i]
			df_merge['s1_insr_pY1361_mixed'].values[i]=df_merge['s1_insr_1361_1'].values[i]
		else:
			df_merge['total_pY1361_mixed'].values[i]=df_merge['richness_1361_0'].values[i]
			df_merge['total_insr_pY1361_mixed'].values[i]=df_merge['richness_insr_1361_0'].values[i]
			df_merge['s1_pY1361_mixed'].values[i]=df_merge['s1_1361_0'].values[i]
			df_merge['s1_insr_pY1361_mixed'].values[i]=df_merge['s1_insr_1361_0'].values[i]
	print(df_merge [(df_merge ['CodeC']==1457) & (df_merge ['CodeB']==210) & (df_merge ['CodeA']==104)])
	df_merge['total_pY1355']=0
	df_merge['total_pY1361']=0
	df_merge['total_insr']=0
	df_merge['s1_pY1355']=0
	df_merge['s1_pY1361']=0
	df_merge['s1_insr']=0
	for col in df_merge.columns:
		if "pY1355.erh_S1" in col:
			df_merge['s1_pY1355'] = df_merge['s1_pY1355']+df_merge[str(col)]
		elif "pY1355.erh" in col:
			df_merge['total_pY1355'] = df_merge['total_pY1355']+df_merge[str(col)]
		elif "pY1361.erh_S1" in col:
			df_merge['s1_pY1361'] = df_merge['s1_pY1361']+df_merge[str(col)]
		elif "pY1361.erh" in col:
			df_merge['total_pY1361'] = df_merge['total_pY1361']+df_merge[str(col)]
		elif "INSR.erh_S1" in col:
			df_merge['s1_insr'] = df_merge['s1_insr']+df_merge[str(col)]
		elif "INSR.erh" in col:
			df_merge['total_insr'] = df_merge['total_insr']+df_merge[str(col)]
	print(df_merge [(df_merge ['CodeC']==1457) & (df_merge ['CodeB']==210) & (df_merge ['CodeA']==104)])

	cols=['richness_1355_0', 'richness_1355_1', 'richness_1361_0', 'richness_1361_1', 'richness_insr_1355_0', 'richness_insr_1355_1', 'richness_insr_1361_0', 'richness_insr_1361_1', 's1_1355_0', 's1_1355_1', 's1_1361_0', 's1_1361_1', 's1_insr_1355_0', 's1_insr_1355_1', 's1_insr_1361_0', 's1_insr_1361_1']
	for col in df_merge.columns:
		if '.erh' in col:
			cols.append(col)
		cols.append('class')
	print('columns to be dropped are: ',cols)
	df_erh_insr =df_merge.drop(cols,axis=1)
	print(df_erh_insr.columns)
	print(df_merge)

	if len(samples)>0:
		for i in range(len(samples)):
			codeA=samples['CodeA'][i]
			codeB=samples['CodeB'][i]
			codeC=samples['CodeC'][i]
			print(df_erh_insr [(df_erh_insr ['CodeC']==codeC) & (df_erh_insr  ['CodeB']==codeB) & (df_erh_insr  ['CodeA']==codeA)])
	return df_erh_insr
def erhAnalysis_tpor(erh_dir,df_filtered,samples):
	print(len(df_filtered))
	df_merge = df_filtered

	array = [[1,2],[3,4],[5,6],[7,8]]

	col_626=[]
	col_tpor=[]
	#df_tpor_total=pd.DataFrame()
	for r,d,f in os.walk(erh_dir):
		print(r)
		for m in range(len(array)):
			num_626 =array[m][0]
			num_tpor =array[m][1]
			for f_ in f:
				if num_626 == int(f_.split("_",2)[-2][-1]):
					print('626 col file is: ',f_)
					col_626.append(f_)
				elif num_tpor == int(f_.split("_",2)[-2][-1]):
					print('TPOR col file is: ',f_)
					col_tpor.append(f_)
		print(col_626)
		print(col_tpor)
		
		for i in range(len(f)):
			f_ = f[i]
			if f_.split(".")[1]=="erh":
				if f_ in col_626:
					print('pY626 file is: ',f_)

					df_626 = pd.read_csv(r+f_,names=['CodeA','CodeB','CodeC','S1','Richness'] )
					df_626 = df_626.sort_values(by=['CodeA','CodeB','CodeC'])
					col_name = str(f_)
					df_626[col_name] = df_626['Richness']
					df_626  = df_626.drop("Richness",axis=1)
					df_626[col_name+'_S1'] = df_626['S1']
					df_626 = df_626.drop("S1",axis=1)
					df_merge = df_merge.merge(df_626,how='left',on=['CodeA','CodeB','CodeC'])
				elif f_ in col_tpor:
					print('TPOR file is: ',f_)
					df_tpor = pd.read_csv(r+f_,names=['CodeA','CodeB','CodeC','S1','Richness'])
					df_tpor = df_tpor.sort_values(by=['CodeA','CodeB','CodeC'])
					print(df_tpor [(df_tpor ['CodeC']==42) & (df_tpor ['CodeB']==703) & (df_tpor ['CodeA']==327)])
					col_name = str(f_)
					df_tpor[col_name] = df_tpor['Richness']
					df_tpor = df_tpor.drop("Richness",axis=1)
					df_tpor[col_name+'_S1'] = df_tpor['S1']
					df_tpor = df_tpor.drop("S1",axis=1)
					#df_tpor_total=pd.concat((df_tpor_total,df_tpor),axis=1)
					df_merge = df_merge.merge(df_tpor,how='left',on=['CodeA','CodeB','CodeC'])
					print(df_merge [(df_merge ['CodeC']==42) & (df_merge ['CodeB']==703) & (df_merge ['CodeA']==327)])
	#df_tpor_total.to_csv('tpor.csv')
	df_merge.fillna(0,inplace=True)
	print(df_merge [(df_merge ['CodeC']==1457) & (df_merge ['CodeB']==210) & (df_merge ['CodeA']==104)])
	df_col = pd.DataFrame()
	df_col['col_626']=col_626
	df_col['col_tpor']=col_tpor
	print(df_col)
	
	df_merge['performance_ind_0_total']=0
	df_merge['richness_626_0']=0
	df_merge['richness_626_1']=0
	df_merge['performance_ind_1_total']=0
	df_merge['richness_tpor_626_0']=0
	df_merge['richness_tpor_626_1']=0
	df_merge['s1_626_0']=0
	df_merge['s1_626_1']=0
	df_merge['s1_tpor_626_0']=0
	df_merge['s1_tpor_626_1']=0


	for i in range(len(df_merge)):
		richiness_626_1 = 0.00
		richiness_626_0 = 0.00
		richiness_tpor_626_1 = 0.00
		richiness_tpor_626_0 = 0.00
		s1_626_1 = 0.00
		s1_626_0 = 0.00
		s1_tpor_626_1 = 0.00
		s1_tpor_626_0 = 0.00
		performance_total_626 = 0
		for j in range(len(df_col)):
			col_626_ = df_col['col_626'].values[j]
			col_tpor_ = df_col['col_tpor'].values[j]
			col_626_S1 = df_col['col_626'].values[j]+'_S1'
			col_tpor_S1 = df_col['col_tpor'].values[j]+'_S1'
			#print(col_626_,col_tpor_)   
			if df_merge[col_626_].values[i]>=df_merge[col_tpor_].values[i]:
				richiness_626_1+=df_merge[col_626_].values[i]
				richiness_tpor_626_1+=df_merge[col_tpor_].values[i]
				s1_626_1+=df_merge[col_626_S1].values[i]
				s1_tpor_626_1+=df_merge[col_tpor_S1].values[i]
				performance_total_626+=1
			else:
				richiness_626_0+=df_merge[col_626_].values[i]
				performance_total_626+=0
				richiness_tpor_626_0+=df_merge[col_tpor_].values[i]
				s1_626_0+=df_merge[col_626_S1].values[i]
				s1_tpor_626_0+=df_merge[col_tpor_S1].values[i]
			#print(col_1361_,col_tpor_)   

		df_merge['performance_ind_0_total'].values[i]=performance_total_626
		df_merge['richness_626_0'].values[i]=richiness_626_0
		df_merge['richness_626_1'].values[i]=richiness_626_1

		df_merge['richness_tpor_626_0'].values[i]=richiness_tpor_626_0
		df_merge['richness_tpor_626_1'].values[i]=richiness_tpor_626_1

		df_merge['s1_626_0'].values[i]=s1_626_0
		df_merge['s1_626_1'].values[i]=s1_626_1

		df_merge['s1_tpor_626_0'].values[i]=s1_tpor_626_0
		df_merge['s1_tpor_626_1'].values[i]=s1_tpor_626_1

	print(df_merge [(df_merge ['CodeC']==1457) & (df_merge ['CodeB']==210) & (df_merge ['CodeA']==104)])
	df_merge['total_pY626_mixed']=0

	df_merge['total_tpor_pY626_mixed']=0

	df_merge['s1_pY626_mixed']=0

	df_merge['s1_tpor_pY626_mixed']=0

	for i in range(len(df_merge)):
		if df_merge['performance_ind_0_total'].values[i]>=3:
			df_merge['total_pY626_mixed'].values[i]=df_merge['richness_626_1'].values[i]
			df_merge['total_tpor_pY626_mixed'].values[i]=df_merge['richness_tpor_626_1'].values[i]
			df_merge['s1_pY626_mixed'].values[i]=df_merge['s1_626_1'].values[i]
			df_merge['s1_tpor_pY626_mixed'].values[i]=df_merge['s1_tpor_626_1'].values[i]
		else:
			df_merge['total_pY626_mixed'].values[i]=df_merge['richness_626_0'].values[i]
			df_merge['total_tpor_pY626_mixed'].values[i]=df_merge['richness_tpor_626_0'].values[i]
			df_merge['s1_pY626_mixed'].values[i]=df_merge['s1_626_0'].values[i]
			df_merge['s1_tpor_pY626_mixed'].values[i]=df_merge['s1_tpor_626_0'].values[i]

	print(df_merge [(df_merge ['CodeC']==1457) & (df_merge ['CodeB']==210) & (df_merge ['CodeA']==104)])
	df_merge['total_pY626']=0
	df_merge['total_tpor']=0
	df_merge['s1_pY626']=0
	df_merge['s1_tpor']=0
	for col in df_merge.columns:
		if "pY626.erh_S1" in col:
			df_merge['s1_pY626'] = df_merge['s1_pY626']+df_merge[str(col)]
		elif "pY626.erh" in col:
			df_merge['total_pY626'] = df_merge['total_pY626']+df_merge[str(col)]
		elif "tpor.erh_S1" in col:
			df_merge['s1_tpor'] = df_merge['s1_tpor']+df_merge[str(col)]
		elif "tpor.erh" in col:
			df_merge['total_tpor'] = df_merge['total_tpor']+df_merge[str(col)]
	print(df_merge [(df_merge ['CodeC']==1457) & (df_merge ['CodeB']==210) & (df_merge ['CodeA']==104)])

	cols=['richness_626_0', 'richness_626_1',  'richness_tpor_626_0', 'richness_tpor_626_1', 's1_626_0', 's1_626_1', 's1_tpor_626_0', 's1_tpor_626_1']
	for col in df_merge.columns:
		if '.erh' in col:
			cols.append(col)
		cols.append('class')
	print('columns to be dropped are: ',cols)
	df_erh_tpor =df_merge.drop(cols,axis=1)
	print(df_erh_tpor.columns)

	if len(samples)>0:
		for i in range(len(samples)):
			codeA=samples['CodeA'][i]
			codeB=samples['CodeB'][i]
			codeC=samples['CodeC'][i]
			print(df_erh_tpor [(df_erh_tpor ['CodeC']==codeC) & (df_erh_tpor  ['CodeB']==codeB) & (df_erh_tpor  ['CodeA']==codeA)])
	return df_erh_tpor

