import pandas as pd
import numpy as np
import datetime
from scipy import stats as st
import matplotlib.pyplot as plt
from models import ipca
import argparse
import gc





def score(df_similarity,opt,samples):
	gc.collect()
	model_names = ['kmeans','agglmerativeclustering']

	print(df_similarity)


	#df_similarity['Similarity_pY1355']=1/abs( df_similarity['total_insr_pY1355_mixed_ipca']-x_min_1355)
	#df_similarity['Similarity_pY1361']=1/abs( df_similarity['total_insr_pY1361_mixed_ipca']-x_min_1361)

	
	print('-------------------filtered dataframe is:------------------------')
	print(df_similarity)
	print(len(df_similarity))

	if len(samples)>0:
		for i in range(len(samples)):
			codeA=samples[i][0]
			codeB=samples[i][1]
			codeC=samples[i][2]
			print(df_similarity [(df_similarity ['CodeC']==codeC) & (df_similarity ['CodeB']==codeB) & (df_similarity ['CodeA']==codeA)])
	
	#gc.collect()

	#df_similarity = pd.read_csv('/Users/xiaotan/Documents/chemistry/data/phase_1/result_kmeans_insr.csv')

	#df_similarity['performance_ind']=max(df_similarity['performance_ind_0_total'],df_similarity['performance_ind_1_total'])
	#df_similarity ['Similarity'] = (df_similarity ['Similarity'].pow(max(df_similarity['performance_ind_0_total'],df_similarity['performance_ind_1_total']).apply(np.exp)))*df_similarity ['S1_Richness_efficiency']*df_similarity ['S1_Richness_balance']
	#df_similarity ['Similarity'] = df_similarity ['Similarity']*df_similarity ['S1_Richness_balance']
	#df_similarity_weighted = df_similarity[df_similarity['S1_Richness_efficiency']>average_efficiency]
	#df_similarity ['Similarity'] = (df_similarity ['Similarity']*abs(df_similarity ['Richness_STDEV']+df_similarity ['S1_STDEV']))
	

	#df_similarity_0['SCORE_RANK']=df_similarity_0['Score'].rank(ascending=True).astype(float)
	df_similarity_1 = df_similarity[df_similarity['class']==1]
	df_similarity_1= df_similarity_1.sort_values(by =['Similarity'],ascending=True)
	df_similarity_1['Similarity_antiinsr_RANK_pY1355']=df_similarity_1['Similarity_antiinsr_1355'].rank(ascending=True).astype(float)
	df_similarity_1['Similarity_antiinsr_RANK_pY1361']=df_similarity_1['Similarity_antiinsr_1361'].rank(ascending=True).astype(float)
	df_similarity_1['Similarity_RANK_pY1355']=df_similarity_1['Similarity_pY1355'].rank(ascending=True).astype(float)
	df_similarity_1['Similarity_RANK_pY1361']=df_similarity_1['Similarity_pY1361'].rank(ascending=True).astype(float)
	df_similarity_1['Similarity_centralLine_RANK_pY1355']=df_similarity_1['Similarity_centralLine_pY1355'].rank(ascending=True).astype(float)
	df_similarity_1['Similarity_centralLine_RANK_pY1361']=df_similarity_1['Similarity_centralLine_pY1361'].rank(ascending=True).astype(float)
	df_similarity_1['Similarity_point_RANK_pY1355']=df_similarity_1['Similarity_point_pY1355'].rank(ascending=True).astype(float)
	df_similarity_1['Similarity_point_RANK_pY1361']=df_similarity_1['Similarity_point_pY1361'].rank(ascending=True).astype(float)
	df_similarity_1['Similarity_point_RANK_pY1355_mixed']=df_similarity_1['Similarity_point_pY1355_mixed'].rank(ascending=True).astype(float)
	df_similarity_1['Similarity_point_RANK_pY1361_mixed']=df_similarity_1['Similarity_point_pY1361_mixed'].rank(ascending=True).astype(float)
	df_similarity_1['S1_Richness_efficiency_RANK']=df_similarity_1['S1_Richness_efficiency'].rank(ascending=False).astype(float)
	df_similarity_1['S1_Richness_balance_RANK']=df_similarity_1['S1_Richness_balance'].rank(ascending=False).astype(float)
	if opt.model_name.lower() in model_names:
		df_similarity_1['Similarity_RANK']=df_similarity_1['Similarity'].rank(ascending=False).astype(float)
	else:
		df_similarity_1['Similarity_RANK']=df_similarity_1['Similarity'].rank(ascending=True).astype(float)
	df_similarity_1['Richness_SUM_RANK']=df_similarity_1['Richness_SUM'].rank(ascending=False).astype(float)
	df_similarity_1['Richness_COUNT_RANK']=df_similarity_1['Richness_COUNT'].rank(ascending=False).astype(float)
	df_similarity_1['S1_SUM_RANK']=df_similarity_1['S1_SUM'].rank(ascending=False).astype(float)
	df_similarity_1['S1_COUNT_RANK']=df_similarity_1['S1_COUNT'].rank(ascending=False).astype(float)
	#df_similarity_1['Score']=(df_similarity_1['Similarity_RANK']+(df_similarity_1['Similarity_point_RANK_pY1355_mixed']+df_similarity_1['Similarity_point_RANK_pY1361_mixed'])/2).astype(float)
	df_similarity_1['Score']=df_similarity_1['Similarity_RANK']+(df_similarity_1['Richness_COUNT_RANK']+df_similarity_1['Richness_SUM_RANK'])/2
	#df_similarity_1['Score']=(df_similarity_1['Similarity_RANK']+(df_similarity_1['Similarity_antiinsr_RANK_pY1355']+df_similarity_1['Similarity_antiinsr_RANK_pY1361'])).astype(float)
	#df_similarity_1['Score']=df_similarity_1['Similarity_RANK']+(df_similarity_1['Richness_COUNT_RANK']+df_similarity_1['Richness_SUM_RANK'])
	#df_similarity_1['Score']=(df_similarity_1['Similarity_RANK']*0.5+(df_similarity_1['Richness_COUNT_RANK']+df_similarity_1['Richness_SUM_RANK'])/2 +(df_similarity_1['Similarity_RANK_pY1355']+df_similarity_1['Similarity_RANK_pY1361'])/2  ).astype(float)
	#df_similarity_1['Score']=(df_similarity_1['Similarity_RANK']+(df_similarity_1['S1_Richness_balance_RANK']+df_similarity_1['S1_Richness_efficiency_RANK'])*0.5).astype(float)
	#df_similarity_1['Score']=df_similarity_1['Similarity_centralLine']+df_similarity_1['Similarity_RANK']+df_similarity_1['Richness_COUNT_RANK']+df_similarity_1['Richness_SUM_RANK']
	#df_similarity_1['Score']=df_similarity_1['Similarity_RANK_pY1355']+df_similarity_1['Similarity_RANK_pY1361']+df_similarity_1['Similarity_RANK']+(df_similarity_1['Richness_COUNT_RANK']+df_similarity_1['Richness_SUM_RANK'])/2
	#df_similarity_1['Score']=df_similarity_1['Similarity_RANK_pY1355']+df_similarity_1['Similarity_RANK_pY1361']+(df_similarity_1['Richness_COUNT_RANK']+df_similarity_1['Richness_SUM_RANK'])/2        
	'''
	top_efficiency = df_similarity_1.sort_values(by =['S1_Richness_efficiency'],ascending=True).head(100)
	top_balance = df_similarity_1.sort_values(by =['S1_Richness_balance'],ascending=True).head(100)
	for i in range(len(df_similarity_1)):
		for j in range(len(top_efficiency)):
			#print(top_efficiency['S1_Richness_efficiency'].values[j], top_balance['S1_Richness_balance'].values[j])
			if df_similarity_1['S1_Richness_efficiency'].values[i] == top_efficiency['S1_Richness_efficiency'].values[j]:
				df_similarity_1['Score'].values[i] = df_similarity_1['Score'].values[i] + df_similarity_1['S1_Richness_efficiency_RANK'].values[i]
			elif df_similarity_1['S1_Richness_balance'].values[i] == top_balance['S1_Richness_balance'].values[j]:
				df_similarity_1['Score'].values[i]=df_similarity_1['Score'].values[i] + df_similarity_1['S1_Richness_balance_RANK'].values[i]
	'''
	for i in range(len(df_similarity_1)):
		weight = max(df_similarity_1['performance_ind_0_total'].values[i],df_similarity_1['performance_ind_1_total'].values[i])
		#print(np.exp(max(weight,1)),weight)
		#df_similarity_1 ['Score'].values[i] = np.power(df_similarity_1 ['Score'].values[i],max(weight,1))
		
		
		if weight==5:
			df_similarity_1 ['Score'].values[i] = df_similarity_1 ['Score'].values[i]/160
		elif weight==4:
			df_similarity_1 ['Score'].values[i] = df_similarity_1 ['Score'].values[i]/144
		elif weight==3:
			df_similarity_1 ['Score'].values[i] = df_similarity_1 ['Score'].values[i]/32
		elif weight==2:
			df_similarity_1 ['Score'].values[i] = df_similarity_1 ['Score'].values[i]/4
		elif weight==1:
			df_similarity_1 ['Score'].values[i] = df_similarity_1 ['Score'].values[i]/2    


	df_similarity_1['SCORE_RANK']=df_similarity_1['Score'].rank(ascending=True).astype(float)

	df_similarity_0 = df_similarity[df_similarity['class']==0]
	df_similarity_0= df_similarity_0.sort_values(by =['Similarity'],ascending=True)
	df_similarity_0['Similarity_RANK_pY1355']=df_similarity_0['Similarity_pY1355'].rank(ascending=True).astype(float)
	df_similarity_0['Similarity_RANK_pY1355']=df_similarity_0['Similarity_pY1355'].rank(ascending=True).astype(float)
	df_similarity_0['Similarity_RANK_pY1361']=df_similarity_0['Similarity_pY1361'].rank(ascending=True).astype(float)
	df_similarity_0['Similarity_centralLine_RANK_pY1355']=df_similarity_0['Similarity_centralLine_pY1355'].rank(ascending=True).astype(float)        
	df_similarity_0['Similarity_centralLine_RANK_pY1361']=df_similarity_0['Similarity_centralLine_pY1361'].rank(ascending=True).astype(float)
	df_similarity_0['Similarity_point_RANK_pY1355']=df_similarity_0['Similarity_point_pY1355'].rank(ascending=True).astype(float)
	df_similarity_0['Similarity_point_RANK_pY1361']=df_similarity_0['Similarity_point_pY1361'].rank(ascending=True).astype(float)
	df_similarity_0['Similarity_point_RANK_pY1355_mixed']=df_similarity_0['Similarity_point_pY1355_mixed'].rank(ascending=True).astype(float)
	df_similarity_0['Similarity_point_RANK_pY1361_mixed']=df_similarity_0['Similarity_point_pY1361_mixed'].rank(ascending=True).astype(float)
	df_similarity_0['Similarity_antiinsr_RANK_pY1355']=df_similarity_0['Similarity_antiinsr_1355'].rank(ascending=True).astype(float)
	df_similarity_0['Similarity_antiinsr_RANK_pY1361']=df_similarity_0['Similarity_antiinsr_1361'].rank(ascending=True).astype(float)
	df_similarity_0['S1_Richness_efficiency_RANK']=df_similarity_0['S1_Richness_efficiency'].rank(ascending=False).astype(float)
	df_similarity_0['S1_Richness_balance_RANK']=df_similarity_0['S1_Richness_balance'].rank(ascending=False).astype(float)
	if opt.model_name.lower() in model_names:
		df_similarity_0['Similarity_RANK']=df_similarity_0['Similarity'].rank(ascending=False).astype(float)
	else:
		df_similarity_0['Similarity_RANK']=df_similarity_0['Similarity'].rank(ascending=True).astype(float)
	df_similarity_0['Richness_SUM_RANK']=df_similarity_0['Richness_SUM'].rank(ascending=False).astype(float)
	df_similarity_0['Richness_COUNT_RANK']=df_similarity_0['Richness_COUNT'].rank(ascending=False).astype(float)
	df_similarity_0['S1_SUM_RANK']=df_similarity_0['S1_SUM'].rank(ascending=False).astype(float)
	df_similarity_0['S1_COUNT_RANK']=df_similarity_0['S1_COUNT'].rank(ascending=False).astype(float)
	#df_similarity_0['Score']=(df_similarity_0['Similarity_RANK']+(df_similarity_0['Similarity_point_RANK_pY1355_mixed']+df_similarity_0['Similarity_point_RANK_pY1361_mixed'])/2).astype(float)
	#df_similarity_0['Score']=(df_similarity_0['Similarity_RANK']+(df_similarity_0['Similarity_antiinsr_RANK_pY1355']+df_similarity_0['Similarity_antiinsr_RANK_pY1361'])).astype(float)
	#df_similarity_0['Score']=df_similarity_0['Similarity_RANK']+(df_similarity_0['Richness_COUNT_RANK']+df_similarity_0['Richness_SUM_RANK'])
	df_similarity_0['Score']=df_similarity_0['Similarity_RANK']+(df_similarity_0['Richness_COUNT_RANK']+df_similarity_0['Richness_SUM_RANK'])/2
	#df_similarity_0['Score']=df_similarity_0['Similarity_centralLine_RANK']+df_similarity_0['Similarity_RANK']+(df_similarity_0['Richness_COUNT_RANK']+df_similarity_0['Richness_SUM_RANK'])/2
	#df_similarity_0['Score']=(df_similarity_0['Similarity_RANK']+(df_similarity_0['S1_Richness_balance_RANK']+df_similarity_0['S1_Richness_efficiency_RANK'])*0.5).astype(float)
	#df_similarity_0['Score']=((df_similarity_0['Similarity_RANK_pY1355']+df_similarity_0['Similarity_RANK_pY1361'])*0.5+df_similarity_0['Similarity_RANK']).astype(float)       
	#df_similarity_0['Score']=(df_similarity_0['Similarity_RANK']*0.5+(df_similarity_0['Richness_COUNT_RANK']+df_similarity_0['Richness_SUM_RANK'])/2+(df_similarity_0['Similarity_RANK_pY1355']+df_similarity_0['Similarity_RANK_pY1361'])/2).astype(float)
	#df_similarity_0['Score']=df_similarity_0['Similarity_centralLine']+df_similarity_0['Similarity_RANK']+df_similarity_0['Richness_COUNT_RANK']+df_similarity_0['Richness_SUM_RANK']
	#df_similarity_0['Score']=df_similarity_0['Similarity_RANK_pY1355']+df_similarity_0['Similarity_RANK_pY1361']+(df_similarity_0['Richness_COUNT_RANK']+df_similarity_0['Richness_SUM_RANK'])/2
	'''
	top_efficiency = df_similarity_0.sort_values(by =['S1_Richness_efficiency'],ascending=True).head(100)
	top_balance = df_similarity_0.sort_values(by =['S1_Richness_balance'],ascending=True).head(100)
	for i in range(len(df_similarity_0)):
		for j in range(len(top_efficiency)):
			#print(top_efficiency['S1_Richness_efficiency'].values[j], top_balance['S1_Richness_balance'].values[j])
			if df_similarity_0['S1_Richness_efficiency'].values[i] == top_efficiency['S1_Richness_efficiency'].values[j]:
				df_similarity_0['Score'].values[i] = df_similarity_0['Score'].values[i] + df_similarity_0['S1_Richness_efficiency_RANK'].values[i]
			elif df_similarity_0['S1_Richness_balance'].values[i] == top_balance['S1_Richness_balance'].values[j]:
				df_similarity_0['Score'].values[i]=df_similarity_0['Score'].values[i] + df_similarity_0['S1_Richness_balance_RANK'].values[i]
	'''
	for i in range(len(df_similarity_0)):
		
		weight = max(df_similarity_0['performance_ind_0_total'].values[i],df_similarity_0['performance_ind_1_total'].values[i])
		#print(np.exp(max(weight,1)),weight)
		#df_similarity_0 ['Score'].values[i] = np.power(df_similarity_0 ['Score'].values[i],max(weight,1))
		
		if weight==5:
			df_similarity_0 ['Score'].values[i] = df_similarity_0 ['Score'].values[i]/160
		elif weight==4:
			df_similarity_0 ['Score'].values[i] = df_similarity_0 ['Score'].values[i]/144
		elif weight==3:
			df_similarity_0 ['Score'].values[i] = df_similarity_0 ['Score'].values[i]/32
		elif weight==2:
			df_similarity_0 ['Score'].values[i] = df_similarity_0 ['Score'].values[i]/4
		elif weight==1:
			df_similarity_0 ['Score'].values[i] = df_similarity_0 ['Score'].values[i]/2
				


	df_similarity_0['SCORE_RANK']=df_similarity_0['Score'].rank(ascending=True).astype(float)+len(df_similarity_1)
	df_score_ = pd.concat((df_similarity_0,df_similarity_1),axis=0)
	df_rank = pd.DataFrame()
	df_rank['SCORE_RANK'] = df_score_['SCORE_RANK'].drop_duplicates()
	df_rank['RANK'] = df_rank['SCORE_RANK'].rank(ascending=True).astype(float)
	df_score = df_score_.merge(df_rank,how='left',on=['SCORE_RANK'])
	df_score['SCORE_RANK']=df_score['SCORE_RANK'].rank(ascending=True).astype(float)
	df_score= df_score.sort_values(by =['SCORE_RANK'],ascending=True)
	print('-------------------filtered dataframe is:------------------------')
	print(df_score)
	print(len(df_score))
	df_score=df_score.reset_index()
	if len(samples)>0:
		for i in range(len(samples)):
			codeA=samples[i][0]
			codeB=samples[i][1]
			codeC=samples[i][2]
			print(df_score [(df_score ['CodeC']==codeC) & (df_score ['CodeB']==codeB) & (df_score ['CodeA']==codeA)])
	df_sample=pd.DataFrame()
	if len(samples)>0:
		for i in range(len(samples)):
			codeA=samples[i][0]
			codeB=samples[i][1]
			codeC=samples[i][2]
			df_sub=df_score [(df_score ['CodeC']==codeC) & (df_score ['CodeB']==codeB) & (df_score ['CodeA']==codeA)]
			df_sample=pd.concat((df_sample,df_sub),axis=0)
	

	df_sample.to_csv('sample_20241017.csv')
	#print(dist)
	result_name = 'output_'+opt.model_name+'_level1.csv'
	df_score.to_csv(opt.output_dir+result_name)

	return df_score