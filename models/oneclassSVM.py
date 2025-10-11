from sklearn.cluster import DBSCAN,KMeans,AgglomerativeClustering,compute_optics_graph,cluster_optics_xi,SpectralClustering,Birch,OPTICS
import shap
from sklearn.svm import SVC,NuSVC,OneClassSVM,SVR,NuSVR
import utils
from utils import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
import datetime
import utils
from utils.utils import searchCV
def model_base():
	global model_base
	model_base = OneClassSVM(kernel='rbf',gamma='auto',nu=0.8,coef0=0.1,tol=1e-5)
	return model_base
def oneclassSVM(x,opt,currentTime,graph_dir):
	model_base=model_base()
	if opt.training_model.lower() == 'gridSearchCV'.lower():
		model = searchCV.searchCV.gridSearchCV(model_base)
	elif opt.training_model.lower() == 'halvingGridSearchCV'.lower():
		model = searchCV.searchCV.halvingGridSearchCV(model_base)
	elif opt.training_model.lower() == None:
		model = model_base
	else:
		print("The parameter training model is not supported for now, please input one of the list  ['None', 'gridSearchCV','halvingGridSearchCV'].")
		exit


	print(model.get_params())

	model = model.fit(x)
	y_predict = model.predict(x)
	score_of_determination = model.score_samples(x)

	x_shap_orig = pd.DataFrame(x,columns=['Enrichment_SUM','Enrichment_STDEV','Enrichment_COUNT','S1_SUM','S1_STDEV','S1_COUNT'])
	x_shap = shap.sample(x_shap_orig,1000)
	x_shap.to_csv('oneClass_x.csv')
	
	explainer_kernel = shap.KernelExplainer(utils.logProbability,x_shap)

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
	shap.summary_plot(shap_values_kernel,x_shap,show=False)        
	fig24=plt.gcf()
	fig24.legend()
	png_name='OCSVM_kernel_summaryplot_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 
	
	plt.figure()



	shap.plots.scatter(shap_values_kernel[:,'Enrichment_SUM'],color=shap_values_kernel[:,'Enrichment_COUNT'],show=False)        
	fig25=plt.gcf()
	fig25.legend()
	png_name='OCSVM_kernel_scatter_Enrichmentsum_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	shap.plots.scatter(shap_values_kernel[:,'S1_SUM'],color=shap_values_kernel,show=False)        
	fig26=plt.gcf()
	fig26.legend()
	png_name='OCSVM_kernel_scatter_s1sum_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	shap.plots.scatter(shap_values_kernel[:,'S1_STDEV'],color=shap_values_kernel,show=False)        
	fig27=plt.gcf()
	fig27.legend()
	png_name='OCSVM_kernel_scatter_s1stdev_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	shap.plots.scatter(shap_values_kernel[:,'Enrichment_STDEV'],color=shap_values_kernel,show=False)        
	fig28=plt.gcf()
	fig28.legend()
	png_name='OCSVM_kernel_scatter_Enrichmentstdev_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	shap.plots.scatter(shap_values_kernel[:,'Enrichment_COUNT'],color=shap_values_kernel,show=False)        
	fig29=plt.gcf()
	fig29.legend()
	png_name='OCSVM_kernel_scatter_Enrichmentcount_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	shap.plots.scatter(shap_values_kernel[:,'S1_COUNT'],color=shap_values_kernel,show=False)        
	fig30=plt.gcf()
	fig30.legend()
	png_name='OCSVM_kernel_scatter_s1count_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 


	plt.figure()
	png_name='OCSVM_kernel_forceplot_'+str(currentTime)+'.png'

	shap.force_plot(np.round(expected_value_kernel,4),np.round(shap_values_kernel.values[0,:],4),np.round(x_shap.iloc[0,:],4),matplotlib=True,show=False,plot_cmap='DrDb').savefig(graph_dir+png_name)  
	#plt.show()
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



	shap.plots.scatter(shap_values[:,'Enrichment_SUM'],color=shap_values[:,'Enrichment_COUNT'],show=False)       
	fig19=plt.gcf()
	fig19.legend()
	png_name='OCSVM_scatter_Enrichmentsum_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	shap.plots.scatter(shap_values[:,'S1_SUM'],color=shap_values,show=False)        
	fig31=plt.gcf()
	fig31.legend()
	png_name='OCSVM_scatter_s1sum_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	shap.plots.scatter(shap_values[:,'S1_STDEV'],color=shap_values,show=False)        
	fig32=plt.gcf()
	fig32.legend()
	png_name='OCSVM_scatter_s1stdev_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	shap.plots.scatter(shap_values[:,'Enrichment_STDEV'],color=shap_values,show=False)        
	fig33=plt.gcf()
	fig33.legend()
	png_name='OCSVM_scatter_Enrichmentstdev_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	shap.plots.scatter(shap_values[:,'Enrichment_COUNT'],color=shap_values,show=False)        
	fig34=plt.gcf()
	fig34.legend()
	png_name='OCSVM_scatter_Enrichmentcount_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	shap.plots.scatter(shap_values[:,'S1_COUNT'],color=shap_values,show=False)        
	fig35=plt.gcf()
	fig35.legend()
	png_name='OCSVM_scatter_s1count_'+str(currentTime)+'.png'
	plt.savefig(graph_dir+png_name)
	plt.close() 

	y_predict[y_predict==1]=0
	y_predict[y_predict==-1]=1

	print('Score of model is: ',score_of_determination)


	return y_predict,score_of_determination