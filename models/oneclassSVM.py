from sklearn.cluster import DBSCAN,KMeans,AgglomerativeClustering,compute_optics_graph,cluster_optics_xi,SpectralClustering,Birch,OPTICS
import shap
from sklearn.svm import SVC,NuSVC,OneClassSVM,SVR,NuSVR
import utils
from utils import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models import Config
import gc
import datetime
def oneclassSVM(x,opt,currentTime):
	global model
	gc.collect()
	graph_dir = opt.graph_dir
	config = Config.OCS_config()
	if opt.parameter_optimizer.lower()=='yes':
		if opt.ocm_optimizer.lower()=='silhouette':
			nu_best,coef_best = utils.optimizeOCS_silhouette(x)
			model = OneClassSVM(**config,nu=nu_best,coef0=coef_best)
		elif opt.ocm_optimizer.lower()=='dense_optmizer':
			nu_best,coef_best = utils.optimizeOCS_dense(x)
			model = OneClassSVM(**config,nu=nu_best,coef0=coef_best)
		else:
			'No support for the optimizer. Default setting is taken.'
			model = OneClassSVM(**config,nu=0.8,coef0=0.1)
	else:
		model = OneClassSVM(**config,nu=0.8,coef0=0.1)
	model = model.fit(x)
	y_predict = model.predict(x)
	score_of_determination = model.score_samples(x)
	'''
	print(model.get_params())

	----------supervised searchcv optimization----------
	model = utils.searchCV.gridSearchCV(model)
	model = model.fit(x_train,y_train)
	----------------------------------------------------
	'''


	df_x = pd.DataFrame(x,columns=['Enrichment_SUM','Enrichment_STDEV','Enrichment_COUNT','S1_SUM','S1_STDEV','S1_COUNT'])
	x_shap_orig = pd.DataFrame()
	x_shap_orig[r'$S_R$']=df_x['Enrichment_SUM']
	x_shap_orig[r'$\sigma_R$']=df_x['Enrichment_STDEV']
	x_shap_orig[r'$S_{nR}$']=df_x['Enrichment_COUNT']
	x_shap_orig[r'$S_C$']=df_x['S1_SUM']
	x_shap_orig[r'$\sigma_C$']=df_x['S1_STDEV']
	x_shap_orig[r'$S_{nC}$']=df_x['S1_COUNT']
	total_samples = len(x_shap_orig)-2
	print(len(x_shap_orig))
	print(len(df_x))
	if len(x_shap_orig)>=total_samples:
		x_shap = shap.sample(x_shap_orig,total_samples)
	else:
		print('total length is less than that of samples')
		x_shap = shap.sample(x_shap_orig)
	x_shap.to_csv('oneClass_x.csv')
	print(x_shap.columns)
	

	explainer_kernel = shap.KernelExplainer(logProbability,x_shap)

	shap_values_kernel=explainer_kernel(x_shap)
	expected_value_kernel = explainer_kernel.expected_value

	plt.figure()

	shap.plots.waterfall(shap_values_kernel[0],show=False)        
	fig22=plt.gcf()
	fig22.legend()
	png_name='OCSVM_kernel_waterfall_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	plt.figure()
	png_name='OCSVM_kernel_forceplot_'+str(currentTime)+'.png'

	shap.force_plot(np.round(expected_value_kernel,4),np.round(shap_values_kernel.values[0,:],4),np.round(x_shap.iloc[0,:],4),matplotlib=True,show=False,plot_cmap='DrDb',contribution_threshold=0.0005).savefig(graph_dir+png_name)  
	plt.show()
	#plt.savefig(graph_dir+png_name)
	plt.close()
	
	'''



	plt.figure()
	shap.summary_plot(shap_values_kernel,x_shap,show=False)        
	fig24=plt.gcf()
	fig24.legend()
	png_name='OCSVM_kernel_summaryplot_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 
	
	plt.figure()



	shap.plots.scatter(shap_values_kernel[:,'$S_R$'],color=shap_values_kernel[:, '$S_{nR}$'],show=False)        
	fig25=plt.gcf()
	fig25.legend()
	png_name='OCSVM_kernel_scatter_Enrichmentsum_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	shap.plots.scatter(shap_values_kernel[:,'$S_C$'],color=shap_values_kernel,show=False)        
	fig26=plt.gcf()
	fig26.legend()
	png_name='OCSVM_kernel_scatter_s1sum_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	shap.plots.scatter(shap_values_kernel[:, '$\sigma_C$'],color=shap_values_kernel,show=False)        
	fig27=plt.gcf()
	fig27.legend()
	png_name='OCSVM_kernel_scatter_s1stdev_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	shap.plots.scatter(shap_values_kernel[:,'$\sigma_R$'],color=shap_values_kernel,show=False)        
	fig28=plt.gcf()
	fig28.legend()
	png_name='OCSVM_kernel_scatter_Enrichmentstdev_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	shap.plots.scatter(shap_values_kernel[:, '$S_{nR}$'],color=shap_values_kernel,show=False)        
	fig29=plt.gcf()
	fig29.legend()
	png_name='OCSVM_kernel_scatter_Enrichmentcount_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	shap.plots.scatter(shap_values_kernel[:,'$S_{nC}$'],color=shap_values_kernel,show=False)        
	fig30=plt.gcf()
	fig30.legend()
	png_name='OCSVM_kernel_scatter_s1count_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 


	plt.figure()

	shap.plots.heatmap(shap_values_kernel,show=False)        
	fig23=plt.gcf()
	fig23.legend()
	png_name='OCSVM_kernel_heatmap_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 
	

	model_orig = OneClassSVM(kernel='rbf',gamma='auto',nu=0.8,coef0=0.1,tol=1e-5)
	model_shap = model_orig.fit(x_shap)
	explainer = shap.Explainer(model_shap.predict,x_shap)
	shap_values=explainer(x_shap)
	print(shap_values)
	print(shap_values[0])

	plt.figure()
	#plt.title('Clustering') 
	shap.plots.waterfall(shap_values[0],show=False)        
	fig18=plt.gcf()
	fig18.legend()
	png_name='OCSVM_waterfall_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	plt.figure()
	shap.plots.force(explainer.expected_value,shap_values,matplotlib=True,show=False,plot_cmap='DrDb')
	fig36=plt.gcf()
	fig36.legend()
	png_name='OCSVM_kernel_forceplot_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	plt.figure()

	shap.plots.heatmap(shap_values,show=False)        
	fig20=plt.gcf()
	fig20.legend()
	png_name='OCSVM_heatmap_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	plt.figure()
	shap.summary_plot(shap_values,x_shap,show=False)        
	fig21=plt.gcf()
	fig21.legend()
	png_name='OCSVM_summaryplot_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	plt.figure()


	shap.plots.scatter(shap_values[:,'$S_R$'],color=shap_values[:,'$S_{nR}$'],show=False)       
	fig19=plt.gcf()
	fig19.legend()
	png_name='OCSVM_scatter_Enrichmentsum_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	shap.plots.scatter(shap_values[:,'$S_C$'],color=shap_values,show=False)        
	fig31=plt.gcf()
	fig31.legend()
	png_name='OCSVM_scatter_s1sum_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	shap.plots.scatter(shap_values[:,'$\sigma_C$'],color=shap_values,show=False)        
	fig32=plt.gcf()
	fig32.legend()
	png_name='OCSVM_scatter_s1stdev_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	shap.plots.scatter(shap_values[:,'$\sigma_R$'],color=shap_values,show=False)        
	fig33=plt.gcf()
	fig33.legend()
	png_name='OCSVM_scatter_Enrichmentstdev_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	shap.plots.scatter(shap_values[:,'$S_{nR}$'],color=shap_values,show=False)        
	fig34=plt.gcf()
	fig34.legend()
	png_name='OCSVM_scatter_Enrichmentcount_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	shap.plots.scatter(shap_values[:,'$S_{nC}$'],color=shap_values,show=False)        
	fig35=plt.gcf()
	fig35.legend()
	png_name='OCSVM_scatter_s1count_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 
	'''
	y_predict[y_predict==1]=0
	y_predict[y_predict==-1]=1

	print('Score of model is: ',score_of_determination)


	return y_predict,score_of_determination

def logProbability(x):

	score = model.score_samples(x)
	score[score==0]=1
	score_logarithm = np.log(score)

	return score_logarithm