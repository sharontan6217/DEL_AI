import numpy as np
import pandas as pd
import models

def classification(df_filtered,opt,samples,currentTime):
	x = np.array(df_filtered[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']])
	if opt.model_name.lower()=='oneclasssvm':
		y_predict,score_of_determination = models.oneclassSVM.oneclassSVM(x,opt,currentTime)
	elif opt.model_name.lower()=='kmeans':
		y_predict  = models.clustering.kmeans(df_filtered,opt)
	elif opt.model_name.lower()=='spectral':
		y_predict  = models.clustering.spectral(df_filtered,opt)
	elif opt.model_name.lower()=='birch':
		y_predict  = models.clustering.birch(df_filtered,opt)
	elif opt.model_name.lower()=='agglomerativeclustering':
		y_predict  = models.clustering.agglomerativeClustering(df_filtered,opt)
	elif opt.model_name.lower()=='opticsclustering':
		y_predict  = models.clustering.opticsClustering(df_filtered,opt)
	else:
		print('Your model is not supported or you input the wrong model name.')
		pass
	df_result=pd.DataFrame()
	df_result['CodeA']=df_filtered['CodeA']
	df_result['CodeB']=df_filtered['CodeB']
	df_result['CodeC']=df_filtered['CodeC']
	df_result['Score_OSVM'] = score_of_determination
	df_result['class'] = y_predict
	df_result = df_result.sort_values(by =['Score_OSVM'],ascending=True)
	df_result['SCORE_RANK_OSVM']=df_result['Score_OSVM'].rank(ascending=True)

	if len(samples)>0:
		for i in range(len(samples)):
			codeA=samples['CodeA'][i]
			codeB=samples['CodeB'][i]
			codeC=samples['CodeC'][i]
			print(df_result [(df_result ['CodeC']==codeC) & (df_result ['CodeB']==codeB) & (df_result ['CodeA']==codeA)])
	
	result_name='output_'+opt.model_name+'.csv'
	df_result.to_csv(opt.output_dir+result_name)
	return y_predict,x,df_result 