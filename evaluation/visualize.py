import pandas as pd
import numpy as np
import datetime
from scipy import stats as st
import matplotlib.pyplot as plt
from models import ipca
import argparse






def Visualize(x,df_similarity,y_predict,df_score,samples,graph_dir):
	#scaler=MinMaxScaler()
	#scaler.fit(x)
	#x=scaler.transform(x)
	#x = np.array(df_result[['S1_ind','Richness_ind','S1_Richness_balance','S1_Richness_efficiency']])
	currentTime=str(datetime.datetime.now()).replace(':','_')
	x_ipca_1355 = np.array(df_similarity[['total_insr_pY1355_mixed_ipca','total_pY1355_mixed_ipca']])
	x_ipca_1361 = np.array(df_similarity[['total_insr_pY1361_mixed_ipca','total_pY1361_mixed_ipca']])
	x_erh_1355 = np.array(df_similarity[['total_insr_pY1355_mixed','total_pY1355_mixed']])    
	x_erh_1361 = np.array(df_similarity[['total_insr_pY1361_mixed','total_pY1361_mixed']])   
	x_antiinsr_pY1355_ipca = np.array(df_similarity[['pY1355_ipca_x','pY1355_ipca_y','pY1355_ipca_z']])   
	x_antiinsr_pY1361_ipca = np.array(df_similarity[['pY1361_ipca_x','pY1361_ipca_y','pY1355_ipca_z']])  
	x_antiinsr_pY1355_ipca_mixed = np.array(df_similarity[['pY1355_mixed_ipca_x','pY1355_mixed_ipca_y','pY1355_mixed_ipca_z']])   
	x_antiinsr_pY1361_ipca_mixed = np.array(df_similarity[['pY1361_mixed_ipca_x','pY1361_mixed_ipca_y','pY1361_mixed_ipca_z']]) 
	x_ipca= ipca.ipca(x,2 )
	df_ipca = pd.DataFrame( x_ipca )
	df_ipca['y'] = y_predict
	plt.scatter(df_ipca[df_ipca['y']==1][0],df_ipca[df_ipca['y']==1][1],label='cluster positive',color='red')
	plt.scatter(df_ipca[df_ipca['y']==0][0],df_ipca[df_ipca['y']==0][1],label='cluster negative',color='blue')
	#plt.title('OCSVM Clustering')
	plt.xlabel('PCA1')
	plt.ylabel('PCA2')
	plt.legend(loc='best')
	fig_1=plt.gcf()
	png_name_roc='output_'+opt.model_name+'_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name_roc)
	plt.close()       
    


	df_sample=pd.DataFrame()
	if len(samples)>0:
		for i in range(len(samples)):
			codeA=samples[i][0]
			codeB=samples[i][1]
			codeC=samples[i][2]
			df_sub=df_similarity [(df_similarity ['CodeC']==codeC) & (df_similarity ['CodeB']==codeB) & (df_similarity ['CodeA']==codeA)]
			df_sample=pd.concat((df_sample,df_sub),axis=0)
	else:
		for i in range(10):
			codeA=df_score['CodeA'][i]
			codeB=df_score['CodeB'][i]
			codeC=df_score['CodeC'][i]
			df_sub=df_score [(df_score ['CodeC']==codeC) & (df_score ['CodeB']==codeB) & (df_score ['CodeA']==codeA)]
			df_sample=pd.concat((df_sample,df_sub),axis=0)

	df_sample = df_sample.reset_index()
	print('-----------samples are----------------')
	print(df_sample)
	text_sample=df_sample['CodeA'].astype(str)+'-'+df_sample['CodeB'].astype(str)+'-'+df_sample['CodeC'].astype(str)
	

	

	x_sample_1355 = df_sample['total_insr_pY1355_mixed_ipca']
	y_sample_1355 = df_sample['total_pY1355_mixed_ipca']
	print(df_sample)
	plt.scatter(x_ipca_1355[:,0],x_ipca_1355[:,1], alpha=0.3)
	'''
	for i, (comp,var) in enumerate(zip(ipca_1355.components_,ipca_1355.explained_variance_)):
		comp = comp * var
		print(comp)
		plt.plot([0,comp[0]],[0,comp[1]],label=f'Component{i}',linewidth=5,color=f'C{i+2}')
	'''
	#plt.title('Clustering')
	plt.plot(x_sample_1355,y_sample_1355,'r8')
	plt.xlabel('PCA 1')
	plt.ylabel('PCA 2')
	plt.xlim(-10,10)
	plt.ylim(-1,5)
	for i in range(len(x_sample_1355)):
		plt.text(x_sample_1355[i],y_sample_1355[i]+0.2,text_sample[i],fontsize='x-small')
	plt.legend(loc='best')
	fig_2=plt.gcf()
	png_name_roc='output_ipca_pY1355_mixed_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name_roc)
	plt.close()  

	#x_sample = df_sample['total_insr']
	x_sample_1361 = df_sample['total_insr_pY1361_mixed_ipca']
	y_sample_1361 = df_sample['total_pY1361_mixed_ipca']
	plt.scatter(x_ipca_1361[:,0],x_ipca_1361[:,1], alpha=0.3)
	'''
	for i, (comp,var) in enumerate(zip(ipca_1361.components_,ipca_1361.explained_variance_)):
		comp = comp * var
		plt.plot([0,comp[0]],[0,comp[1]],label=f'Component{i}',linewidth=5,color=f'C{i+2}')
	'''
	#plt.title('Clustering')
	plt.plot(x_sample_1361,y_sample_1361,'r8')
	plt.xlabel('PCA 1')
	plt.ylabel('PCA 2')
	plt.xlim(-10,10)
	plt.ylim(-1,5)
	for i in range(len(x_sample_1361)):
		plt.text(x_sample_1361[i],y_sample_1361[i]+0.2,text_sample[i],fontsize='x-small')
	plt.legend(loc='best')
	fig_3=plt.gcf()
	png_name_roc='output_ipca_pY1361_mixed_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name_roc)
	plt.close()  
	
	x_sample = df_sample['total_insr_pY1355_mixed']
	y_sample_1355 = df_sample['total_pY1355_mixed']
	plt.scatter(x_erh_1355[:,0],x_erh_1355[:,1], alpha=0.3)
	points = [(x,x) for x in range(0,4000)]
	x_line,y_line = zip(*points)
	plt.plot(x_line,y_line)
	plt.xlim(0,4000)
	plt.ylim(0,4000)
	plt.plot(x_sample,y_sample_1355,'r8')
	#plt.title('Clustering')
	plt.xlabel('Enrichment on Anti-INSR')
	plt.ylabel('Enrichment on Anti- pY1355')
	for i in range(len(x_sample)):
		plt.text(x_sample[i],y_sample_1355[i]+0.2,text_sample[i],fontsize='x-small')
	plt.legend(loc='best')
	fig_4=plt.gcf()
	png_name_roc='output_ipca_pY1355_orig_mixed_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name_roc)
	plt.close()  

	x_sample = df_sample['total_insr_pY1361_mixed']
	y_sample_1361 = df_sample['total_pY1361_mixed']
	plt.scatter(x_erh_1361[:,0],x_erh_1361[:,1], alpha=0.3)
	points = [(x,x) for x in range(0,2000)]
	x_line,y_line = zip(*points)
	plt.plot(x_line,y_line)
	#plt.title('Clustering')
	plt.xlim(0,2000)
	plt.ylim(0,2000)
	plt.xlabel('Enrichment on Anti-INSR')
	plt.ylabel('Enrichment on Anti-pY1361')
	plt.plot(x_sample,y_sample_1361,'r8')
	for i in range(len(x_sample)):
		plt.text(x_sample[i],y_sample_1361[i]+0.2,text_sample[i],fontsize='x-small')
	plt.legend(loc='best')
	fig_5=plt.gcf()
	png_name_roc='output_ipca_pY1361_orig_mixed_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name_roc)
	plt.close()  

	x_ipca_1355 = np.array(df_similarity[['total_insr_1355_ipca','total_pY1355_ipca']])
	x_ipca_1361 = np.array(df_similarity[['total_insr_1361_ipca','total_pY1361_ipca']])
	x_erh_1355 = np.array(df_similarity[['total_insr','total_pY1355']])    
	x_erh_1361 = np.array(df_similarity[['total_insr','total_pY1361']])   

	x_sample_1355 = df_sample['total_insr_1355_ipca']
	y_sample_1355 = df_sample['total_pY1355_ipca']
	print(df_sample)
	plt.scatter(x_ipca_1355[:,0],x_ipca_1355[:,1], alpha=0.3)
	'''
	for i, (comp,var) in enumerate(zip(ipca_1355.components_,ipca_1355.explained_variance_)):
		comp = comp * var
		print(comp)
		plt.plot([0,comp[0]],[0,comp[1]],label=f'Component{i}',linewidth=5,color=f'C{i+2}')
	'''
	#plt.title('Clustering')
	plt.plot(x_sample_1355,y_sample_1355,'r8')
	plt.xlabel('PCA 1')
	plt.ylabel('PCA 2')
	for i in range(len(x_sample_1355)):
		plt.text(x_sample_1355[i],y_sample_1355[i]+0.2,text_sample[i],fontsize='x-small')
	plt.legend(loc='best')
	fig_6=plt.gcf()
	png_name_roc='output_ipca_pY1355_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name_roc)
	plt.close()  

	#x_sample = df_sample['total_insr']
	x_sample_1361 = df_sample['total_insr_1361_ipca']
	y_sample_1361 = df_sample['total_pY1361_ipca']
	plt.scatter(x_ipca_1361[:,0],x_ipca_1361[:,1], alpha=0.3)
	'''
	for i, (comp,var) in enumerate(zip(ipca_1361.components_,ipca_1361.explained_variance_)):
		comp = comp * var
		plt.plot([0,comp[0]],[0,comp[1]],label=f'Component{i}',linewidth=5,color=f'C{i+2}')
	'''
	#plt.title('Clustering')
	plt.plot(x_sample_1361,y_sample_1361,'r8')
	plt.xlabel('PCA 1')
	plt.ylabel('PCA 2')
	for i in range(len(x_sample_1361)):
		plt.text(x_sample_1361[i],y_sample_1361[i]+0.2,text_sample[i],fontsize='x-small')
	plt.legend(loc='best')
	fig_7=plt.gcf()
	png_name_roc='output_ipca_pY1361_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name_roc)
	plt.close()  
	
	x_sample = df_sample['total_insr']
	y_sample_1355 = df_sample['total_pY1355']
	plt.scatter(x_erh_1355[:,0],x_erh_1355[:,1], alpha=0.3)
	points = [(x,x) for x in range(0,4000)]
	x_line,y_line = zip(*points)
	plt.plot(x_line,y_line)
	plt.xlim(0,4000)
	plt.ylim(0,4000)
	plt.plot(x_sample,y_sample_1355,'r8')
	#plt.title('Clustering')
	plt.xlabel('Enrichment on INSR')
	plt.ylabel('Enrichment on pY1355')
	for i in range(len(x_sample)):
		plt.text(x_sample[i],y_sample_1355[i]+0.2,text_sample[i],fontsize='x-small')
	plt.legend(loc='best')
	fig_8=plt.gcf()
	png_name_roc='output_ipca_pY1355_orig_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name_roc)
	plt.close()  

	x_sample = df_sample['total_insr']
	y_sample_1361 = df_sample['total_pY1361']
	plt.scatter(x_erh_1361[:,0],x_erh_1361[:,1], alpha=0.3)
	points = [(x,x) for x in range(0,4000)]
	x_line,y_line = zip(*points)
	plt.plot(x_line,y_line)
	#plt.title('Clustering')
	plt.xlim(0,4000)
	plt.ylim(0,4000)
	plt.xlabel('Enrichment on INSR')
	plt.ylabel('Enrichment on pY1355')
	plt.plot(x_sample,y_sample_1361,'r8')
	for i in range(len(x_sample)):
		plt.text(x_sample[i],y_sample_1361[i]+0.2,text_sample[i],fontsize='x-small')
	plt.legend(loc='best')
	fig_9=plt.gcf()
	png_name_roc='output_ipca_pY1361_orig_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name_roc)
	plt.close()  

	x_sample = df_sample['pY1355_ipca_x']
	y_sample = df_sample['pY1355_ipca_y']
	z_sample = df_sample['pY1355_ipca_z']
	ax = plt.axes(projection='3d')
	ax.scatter3D(x_antiinsr_pY1355_ipca[:,0],x_antiinsr_pY1355_ipca[:,1],x_antiinsr_pY1355_ipca[:,2], color='red',alpha=0.3)
	ax.plot(x_sample,y_sample,z_sample,'b8')
	#plt.title('Clustering')
	ax.set_xlabel('PCA 1')
	ax.set_ylabel('PCA 2')
	ax.set_zlabel('PCA 3')
	for i in range(len(x_sample)):
		ax.text(x_sample[i],y_sample[i]+0.2,z_sample[i],text_sample[i],fontsize='x-small')
	plt.legend(loc='best')
	fig_10=plt.gcf()
	png_name_roc='output_ipca_antiinsr_pY1355_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name_roc)
	plt.close()  

	x_sample = df_sample['pY1361_ipca_x']
	y_sample = df_sample['pY1361_ipca_y']
	z_sample = df_sample['pY1361_ipca_z']
	ax = plt.axes(projection='3d')
	ax.scatter3D(x_antiinsr_pY1361_ipca[:,0],x_antiinsr_pY1361_ipca[:,1],x_antiinsr_pY1361_ipca[:,2], color='red',  alpha=0.3)
	ax.plot(x_sample,y_sample,z_sample,'b8')
	#plt.title('Clustering')
	ax.set_xlabel('PCA 1')
	ax.set_ylabel('PCA 2')
	ax.set_zlabel('PCA 3')
	for i in range(len(x_sample)):
		ax.text(x_sample[i],y_sample[i]+0.2,z_sample[i],text_sample[i],fontsize='x-small')
	plt.legend(loc='best')
	fig_11=plt.gcf()
	png_name_roc='output_ipca_antiinsr_pY1361_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name_roc)
	plt.close()  

	x_sample = df_sample['pY1355_mixed_ipca_x']
	y_sample = df_sample['pY1355_mixed_ipca_y']
	z_sample = df_sample['pY1355_mixed_ipca_z']
	ax = plt.axes(projection='3d')
	ax.scatter3D( x_antiinsr_pY1355_ipca_mixed [:,0],x_antiinsr_pY1355_ipca_mixed [:,1],x_antiinsr_pY1355_ipca_mixed [:,2], color='red', alpha=0.3)
	#plt.title('Clustering')
	ax.plot(x_sample,y_sample,z_sample,'b8')
	#plt.title('Clustering')
	ax.set_xlabel('PCA 1')
	ax.set_ylabel('PCA 2')
	ax.set_zlabel('PCA 3')
	for i in range(len(x_sample)):
		ax.text(x_sample[i],y_sample[i]+0.2,z_sample[i],text_sample[i],fontsize='x-small')
	plt.legend(loc='best')
	fig_12=plt.gcf()
	png_name_roc='output_ipca_antiinsr_pY1355_mixed_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name_roc)
	plt.close()  

	x_sample = df_sample['pY1361_mixed_ipca_x']
	y_sample = df_sample['pY1361_mixed_ipca_y']
	z_sample = df_sample['pY1361_mixed_ipca_z']
	ax = plt.axes(projection='3d')
	ax.scatter3D( x_antiinsr_pY1361_ipca_mixed [:,0],x_antiinsr_pY1361_ipca_mixed [:,1],x_antiinsr_pY1361_ipca_mixed [:,2], color='red',  alpha=0.3)
	#plt.title('Clustering')
	ax.plot(x_sample,y_sample,z_sample,'b8')
	#plt.title('Clustering')
	ax.set_xlabel('PCA 1')
	ax.set_ylabel('PCA 2')
	ax.set_zlabel('PCA 3')
	for i in range(len(x_sample)):
		ax.text(x_sample[i],y_sample[i]+0.2,z_sample[i],text_sample[i],fontsize='x-small')
	plt.legend(loc='best')
	fig_13=plt.gcf()
	png_name_roc='output_ipca_antiinsr_pY1361_mixed_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name_roc)
	plt.close()  
	return fig_1,fig_2,fig_3,fig_4,fig_5,fig_6,fig_7,fig_8,fig_9,fig_10,fig_11

