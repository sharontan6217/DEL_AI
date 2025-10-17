import pandas as pd
import numpy as np
import utils
from utils import utils
import gc

def similarity(df_erh_insr_ipca,opt):
	gc.collect()
	model_names = ['kmeans','AgglmerativeClustering']

	df = df_erh_insr_ipca
	print(df.columns)


	y_min_1355 = min(df['total_pY1355_ipca'] )
	y_min_1361 = min(df['total_pY1361_ipca'] )
	x_min_1355 = min(df['total_insr_1355_ipca'])
	x_min_1361= min(df['total_insr_1361_ipca'])
	x_argmin_1355 = df['total_insr_1355_ipca']  [np.argmin(df['total_pY1355_ipca'])]
	x_argmin_1361 = df['total_insr_1361_ipca']  [np.argmin(df['total_pY1361_ipca'])]   
	y_argmin_1355 = df['total_pY1355_ipca']  [np.argmin(df['total_insr_1355_ipca'])]
	y_argmin_1361 = df['total_pY1361_ipca']  [np.argmin(df['total_insr_1361_ipca'])]   
	x_min = min(df['total_insr'])
	print(x_min_1355,y_min_1355)
	print(x_min_1361,y_min_1361 )
	erh_array_1355 = np.array(df[['total_insr_1355_ipca','total_pY1355_ipca']])
	erh_array_1361 = np.array(df[['total_insr_1361_ipca','total_pY1361_ipca']])
	df_xmin_pY1355= df[df['total_insr_1355_ipca']<x_argmin_1355]
	df_xymin_pY1355 = df_xmin_pY1355[df_xmin_pY1355['total_pY1355_ipca']<y_argmin_1355]
	if len(df_xymin_pY1355)>0:
		ct_ind_pY1355= 1
	else:
		ct_ind_pY1355 = 0
	df_xmin_pY1361= df[df['total_insr_1361_ipca']<x_argmin_1361]
	df_xymin_pY1361 = df_xmin_pY1361[df_xmin_pY1361['total_pY1361_ipca']<y_argmin_1361]
	if len(df_xymin_pY1361)>0:
		ct_ind_pY1361= 1
	else:
		ct_ind_pY1361 = 0
	x_min = min(df['total_insr'])

	if ct_ind_pY1355 == 1:
		x_min_1355_ipca = x_argmin_1355 
		df['Similarity_centralLine_pY1355'] = ( df['total_insr_1355_ipca']-x_min_1355_ipca)
		c_1355 = [x_min_1355_ipca,y_min_1355]
		df['Similarity_point_pY1355'] = utils.distanceCP(erh_array_1355,c_1355)
	else:
		y_min_1355_ipca = df['total_pY1355_ipca']  [np.argmin(df['total_insr_1355_ipca'])]
		df['Similarity_centralLine_pY1355'] = ( df['total_pY1355_ipca']-y_min_1355_ipca)
		c_1355 = [x_min_1355,y_min_1355_ipca]
		df['Similarity_point_pY1355'] = utils.distanceCP(erh_array_1355,c_1355)
	if ct_ind_pY1361 == 1:
		x_min_1361_ipca = x_argmin_1361
		df['Similarity_centralLine_pY1361'] = ( df['total_insr_1361_ipca']-x_min_1361_ipca)
		c_1361 = [x_min_1361_ipca,y_min_1361]   
		print(c_1361 )
		df['Similarity_point_pY1361'] = utils.distanceCP(erh_array_1361,c_1361)
	else:
		y_min_1361_ipca = df['total_pY1361_ipca']  [np.argmin(df['total_insr_1361_ipca'])]
		df['Similarity_centralLine_pY1361'] = ( df['total_pY1361_ipca']-y_min_1361_ipca)
		c_1361 = [x_min_1361,y_min_1361_ipca]   
		df['Similarity_point_pY1361'] = utils.distanceCP(erh_array_1361,c_1361)
	df['Similarity_pY1355']=df['Similarity_point_pY1355'] + abs(df['Similarity_centralLine_pY1355'])
	df['Similarity_pY1361']=df['Similarity_point_pY1361'] + abs(df['Similarity_centralLine_pY1361'])

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
	print(x_min_1355_mixed,x_min_1361_mixed,x_argmin_1355_mixed,x_argmin_1361_mixed,y_min_1361_mixed,y_argmin_1361_mixed)
	#df_xmin_pY1355_mixed= df[df['total_insr_pY1355_mixed_ipca']<x_argmin_1355_mixed]
	#print('df_xmin_pY1355_mixed')
	#print(df_xmin_pY1355_mixed)
	#df_ymin_pY1355_mixed = df_xmin_pY1355_mixed[df_xmin_pY1355_mixed['total_pY1355_mixed_ipca']<y_argmin_1355_mixed]
	#print('df_ymin_pY1355_mixed')
	#print(df_ymin_pY1355_mixed)
	if abs(x_min_1355_mixed)>abs(x_argmin_1355_mixed):
		ct_ind_pY1355_mixed= 1
	else:
		ct_ind_pY1355_mixed = 0
	#df_xmin_pY1361_mixed= df[df['total_insr_pY1361_mixed_ipca']<x_argmin_1361_mixed]
	#print('df_xmin_pY1361_mixed')
	#print(df_xmin_pY1361_mixed)
	#df_ymin_pY1361_mixed = df_xmin_pY1361_mixed[df_xmin_pY1361_mixed['total_pY1361_mixed_ipca']<y_argmin_1361_mixed]
	#print('df_ymin_pY1361_mixed')
	#print(df_ymin_pY1361_mixed)
	if abs(x_min_1361_mixed)>abs(x_argmin_1361_mixed):
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

	print(ct_ind_pY1361_mixed,ct_ind_pY1355_mixed,ct_ind_pY1355,ct_ind_pY1361)
	print(abs(x_min_1355_mixed),abs(x_argmin_1355_mixed),abs(y_min_1355_mixed),abs(y_argmin_1355_mixed))

	if opt.level.lower().replace(' ','')=='level1':

		df_similarity = df_similarity[(df_similarity['Similarity_centralLine_pY1355_mixed_ipca']>=0)]




	return df_similarity

def similarity_tpor(df_erh_tpor_ipca,opt):
	gc.collect()
	model_names = ['kmeans','AgglmerativeClustering']

	df = df_erh_tpor_ipca
	print(df.columns)


	y_min_626 = min(df['total_pY626_ipca'] )
	x_min_626 = min(df['total_tpor_626_ipca'])
	x_argmin_626 = df['total_tpor_626_ipca']  [np.argmin(df['total_pY626_ipca'])]
	y_argmin_626 = df['total_pY626_ipca']  [np.argmin(df['total_tpor_626_ipca'])]  
	x_min = min(df['total_tpor'])
	print(x_min_626,y_min_626)
	erh_array_626 = np.array(df[['total_tpor_626_ipca','total_pY626_ipca']])
	df_xmin_pY626= df[df['total_tpor_626_ipca']<x_argmin_626]
	df_xymin_pY626 = df_xmin_pY626[df_xmin_pY626['total_pY626_ipca']<y_argmin_626]
	if len(df_xymin_pY626)>0:
		ct_ind_pY626= 1
	else:
		ct_ind_pY626 = 0


	x_min = min(df['total_tpor'])

	if ct_ind_pY626 == 1:
		x_min_626_ipca = x_argmin_626 
		df['Similarity_centralLine_pY626'] = ( df['total_tpor_626_ipca']-x_min_626_ipca)
		c_626 = [x_min_626_ipca,y_min_626]
		df['Similarity_point_pY626'] = utils.distanceCP(erh_array_626,c_626)
	else:
		y_min_626_ipca = df['total_pY626_ipca']  [np.argmin(df['total_tpor_626_ipca'])]
		df['Similarity_centralLine_pY626'] = ( df['total_pY626_ipca']-y_min_626_ipca)
		c_626 = [x_min_626,y_min_626_ipca]
		df['Similarity_point_pY626'] = utils.distanceCP(erh_array_626,c_626)

	df['Similarity_pY626']=df['Similarity_point_pY626'] + abs(df['Similarity_centralLine_pY626'])


	y_min_626_mixed = min(df['total_pY626_mixed_ipca'])

	x_min_626_mixed = min(df['total_tpor_pY626_mixed_ipca'])


	x_argmin_626_mixed =  df['total_tpor_pY626_mixed_ipca']  [np.argmin(df['total_pY626_mixed_ipca'])]

	y_argmin_626_mixed = df['total_pY626_mixed_ipca']  [np.argmin(df['total_tpor_626_ipca'])]
 

	erh_array_626_mixed = np.array(df[['total_tpor_pY626_mixed_ipca','total_tpor_pY626_mixed_ipca']])

	print(x_min_626_mixed,y_min_626_mixed)
	print(x_min_626_mixed,x_argmin_626_mixed)
	#df_xmin_pY626_mixed= df[df['total_tpor_pY626_mixed_ipca']<x_argmin_626_mixed]
	#print('df_xmin_pY626_mixed')
	#print(df_xmin_pY626_mixed)
	#df_ymin_pY626_mixed = df_xmin_pY626_mixed[df_xmin_pY626_mixed['total_pY626_mixed_ipca']<y_argmin_626_mixed]
	#print('df_ymin_pY626_mixed')
	#print(df_ymin_pY626_mixed)
	if abs(x_min_626_mixed)>abs(x_argmin_626_mixed):
		ct_ind_pY626_mixed= 1
	else:
		ct_ind_pY626_mixed = 0

	

	if ct_ind_pY626_mixed == 1:

		df['Similarity_antitpor_626'] = 1/(df['total_tpor_pY626_mixed_ipca']-x_min_626_mixed)
		y_min_626_mixed_ipca = y_min_626_mixed 
		x_min_626_mixed_ipca = x_argmin_626_mixed 
		df['Similarity_centralLine_pY626_mixed_ipca'] = ( df['total_tpor_pY626_mixed_ipca']-x_min_626_mixed_ipca)
		c_626_mixed = [x_min_626_mixed_ipca,y_min_626_mixed_ipca]
		df['Similarity_point_pY626_mixed'] = utils.distanceCP(erh_array_626_mixed,c_626_mixed)
		print('central line is along fixed x ',  x_min_626_mixed_ipca )
	else:

		df['Similarity_antitpor_626'] = 1/(df['total_pY626_mixed_ipca']-y_min_626_mixed)
		x_min_626_mixed_ipca = x_min_626_mixed
		y_min_626_mixed_ipca = df['total_pY626_mixed_ipca']  [np.argmin(df['total_tpor_pY626_mixed_ipca'])]
			
		df['Similarity_centralLine_pY626_mixed_ipca'] = ( df['total_pY626_mixed_ipca']-y_min_626_mixed_ipca)
		c_626_mixed = [x_min_626_mixed_ipca,y_min_626_mixed_ipca]
		df['Similarity_point_pY626_mixed'] = utils.distanceCP(erh_array_626_mixed,c_626_mixed)
		print('central line is along fixed y ', y_min_626_mixed_ipca )

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
	print(ct_ind_pY626_mixed,ct_ind_pY626)

	print(abs(x_min_626_mixed),abs(x_argmin_626_mixed),abs(y_min_626_mixed),abs(y_argmin_626_mixed))

	return df_similarity
