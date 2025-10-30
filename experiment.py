
import pandas as pd
import numpy as np
import datetime

import models
from models import ipca
import similarityAnalysis
from similarityAnalysis import ipcaAnalysis,similarity
import data
from data import dataLoading_filter,dataLoading_rank,erhAnalysis,erhAnalysis_tpor
from scipy import stats as st
import matplotlib.pyplot as plt
import evaluation
from evaluation import visualize,scoring
import classification
from classification import classification
import torch
import utils
from utils import preprocess, utils
import argparse
import gc
import os
currentTime=str(datetime.datetime.now()).replace(':','_')
device=torch.device("mps")
torch.cuda.empty_cache()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='./data/tpor/total.csv', help = 'directory of the original data' )
    #parser.add_argument('--data_dir',type=str,default='./output/tpor/level_0/output_OneClassSVM_phase1_level0.csv', help = 'directory of the original data' ) # this is for level 1
    #parser.add_argument('--data_dir',type=str,default='./data/phase_1/total.csv', help = 'directory of the original data' )
    #parser.add_argument('--data_dir',type=str,default='./output/phase_1/level_0/output_OneClassSVM_phase1_level0.csv', help = 'directory of the original data' ) # this is for level 1
    #parser.add_argument('--data_dir',type=str,default='./data/phase_2/total.csv', help = 'directory of the original data' )
    #parser.add_argument('--data_dir',type=str,default='C:/Users/sharo/Documents/DEL_AI-main/output_anomaly/phase_2/level_0/output_OneClassSVM_phase2_level0.csv', help = 'directory of the original data' ) # this is for level 1
    parser.add_argument('--erh_dir',type=str,default='C:/Users/sharo/Documents/DEL_AI-main/data/tpor/erh/', help = 'directory of erh files' )
    parser.add_argument('--graph_dir',type=str,default='C:/Users/sharo/Documents/DEL_AI-main/graph/tpor/level_1/', help = 'directory of graphs' )
    parser.add_argument('--output_dir',type=str,default='C:/Users/sharo/Documents/DEL_AI-main/output/tpor/level_1/', help = 'directory of outputs')
    parser.add_argument('--level',type=str,default='level1', help = 'level is either "level0" or "level1". Level0 is for filtering, and level1 is for ranking.')
    parser.add_argument('--model_name',type=str,default='OneClassSVM', help = 'clustering model is one of the list ["OneClassSVM","KMeans","Spectral","BIRCH","AgglomerativeClustering","OpticsClustering"].')
    parser.add_argument('--parameter_optimizer',type=str,default='Yes', help = 'OCSVM model can be auto-optimized.Default to "Yes".')
    parser.add_argument('--ocm_optimizer',type=str,default='dense_optmizer', help = 'OCM optimizer model can be on of the list ["dense_optmizer","silhouette"].')
    parser.add_argument('--amplify_filtering',type=str,default='No', help = 'We add a few amplifications in preporcessing as an option. Input "Yes" if the dataset is large.')
    opt = parser.parse_args()
    return opt

if __name__=='__main__':
    
    project_dir=os.getcwd()
    os.chdir(project_dir)
    gc.collect()
    opt = get_parser()
    data_dir = opt.data_dir
    erh_dir = opt.erh_dir
    graph_dir = opt.graph_dir
    output_dir = opt.output_dir
    model_name=opt.model_name
    samples=pd.read_csv('samples_tpor.csv')
    if os.path.exists(graph_dir)==False:
        os.makedirs(graph_dir)
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)
    #----------------------------Load data-------------------------------------------------------------------------
    if opt.level=='level0':
        df_orig = dataLoading_filter(data_dir,samples)
    else:
        df_orig = dataLoading_rank(data_dir,samples)
    print(len(df_orig))
    #----------------------------Preprocess------------------------------------------------------------------------
    df_orig = preprocess.preprocess.descriptors(df_orig,samples)   
    df_normalized = preprocess.preprocess.dataStandardize(df_orig,samples)                #scale data with standard scaler 
    #----------------------------Binding block-based filterings----------------------------------------------------
    if opt.level=='level0':
        df = preprocess.preprocess.dataPreprocess_filter(df_normalized,samples,opt)           #binding blocks filtering for level 0 to filter the possible active candidates
    else:
        df = preprocess.preprocess.dataPreprocess_rank(df_normalized,samples,opt)             #binding blocks filtering for level 1 to filter the most proactive candidates

    #----------------------------Auto-filtering of outliers with self-adaptive DBSCAN------------------------------

    x_efficiency = np.array(df['S1_Richness_efficiency'])
    x_efficiency = np.reshape(x_efficiency,(len(x_efficiency),1))  
    df_filtered = preprocess.outlierFiltering.outlierFiltering(x_efficiency,df,1,samples)
    x_balance= np.array(df_filtered['S1_Richness_balance'])
    x_balance = np.reshape(x_balance,(len(x_balance),1))
    df_filtered =  preprocess.outlierFiltering.outlierFiltering(x_balance,df_filtered,2,samples)
    #x_dual = np.array(df_filtered[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']])
    #df_filtered = outlierFiltering(x_dual,df_filtered,3)
    #x_bind_0 = np.array(df_filtered[['S1_ind','Richness_ind','S1_Richness_balance','S1_Richness_efficiency']])
    #df_filtered = outlierFiltering(x_bind_0,df_filtered,1)
    x_bind_1 = np.array(df_filtered[['S1_ind','Richness_ind']])
    df_filtered =  preprocess.outlierFiltering.outlierFiltering(x_bind_1,df_filtered,3,samples)
    #x_bind_2 = np.array(df_filtered[['S1_STDEV','Richness_STDEV']])
    #df_filtered = preprocess.outlierFiltering(x_bind_2,df_filtered,4)

    #----------------------------Add INSR comparison indicator and conditional summation---------------------------

    #df_erh_insr = erhAnalysis(erh_dir,df_filtered,samples)
    df_erh_tpor = erhAnalysis_tpor(erh_dir,df_filtered,samples)
    
    #----------------------------OCSVM for classification----------------------------------------------------------
    y_predict,x,df_result =classification(df_erh_tpor,opt,samples,currentTime)

    #----------------------------IPCA for Visulization and Hit Candidate Clustering--------------------------------
    df_erh_tpor_ipca = ipcaAnalysis.ipcaAnalysis_tpor(df_erh_tpor,df_result)
    #df_erh_insr_ipca = ipcaAnalysis.ipcaAnalysis(df_erh_insr,df_result)
    #----------------------------Similarity and Ranking------------------------------------------------------------
    df_similarity = similarity.similarity_tpor(df_erh_tpor_ipca,opt)
    #df_similarity = similarity.similarity(df_erh_insr_ipca,opt)
    #df_score = scoring.score(df_similarity,opt,samples )
    df_score = scoring.score_tpor(df_similarity,opt,samples )
    #----------------------------Generate graphs-------------------------------------------------------------------
    #fig_1,fig_2,fig_3,fig_4,fig_5,fig_6,fig_7,fig_8,fig_9,fig_10,fig_11= visualize.Visualize(x,df_similarity,y_predict,df_score,samples,opt)
    fig_1,fig_2,fig_4,fig_6,fig_8,fig_10= visualize.Visualize_tpor(x,df_similarity,y_predict,df_score,samples,opt)
    
    


    del df_result
    del df_erh_insr
    #del df_erh_tpor
    del df_filtered
    del df_similarity
    del df
