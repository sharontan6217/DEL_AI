
import pandas as pd
import numpy as np
import datetime

import models
from models import ipca
import similarityAnalysis
from similarityAnalysis import ipcaAnalysis,similarity
import data
from data import dataLoading_filter,dataLoading_rank,erhAnalysis
from scipy import stats as st
import matplotlib.pyplot as plt
import evaluation
from evaluation import visualize,scoring
import classification
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
    parser.add_argument('--data_dir',type=str,default='./data/phase_1/rs/total.csv', help = 'directory of the original data' )
    parser.add_argument('--erh_dir',type=str,default='./data/phase_1/erh/', help = 'directory of erh files' )
    parser.add_argument('--graph_dir',type=str,default='./graph/phase_1/level_0/', help = 'directory of graphs' )
    parser.add_argument('--output_dir',type=str,default='./output/phase_1/level_0/', help = 'directory of outputs')
    parser.add_argument('--level',type=str,default='level1', help = 'level is either "level0" or "level1". Level0 is for filtering, and level1 is for ranking.')
    parser.add_argument('--model_name',type=str,default='OneClassSVM', help = 'clustering model is one of the list ["OneClassSVM","KMeans","Spectral","BIRCH","AgglomerativeClustering","OpticsClustering"].')
    parser.add_argument('--amplify_deviation_filtering',type=str,default='No', help = 'We add a few amplifications in preporcessing as an option. Input "Yes" if the dataset is large.')
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
    samples=pd.read_csv('samples_phase1.csv')
    #----------------------------Load data-------------------------------------------------------------------------
    df_orig = dataLoading_filter(data_dir)

    #----------------------------Preprocess------------------------------------------------------------------------
    df_orig = preprocess.descriptors(df_orig,samples)                          #calculate statistic descriptors
    df_normalized = preprocess.dataStandardize(df_orig,samples)                #scale data with standard scaler 
    #----------------------------Binding block-based filterings----------------------------------------------------
    if opt.level=='level0':
        df = preprocess.dataPreprocess_filter(df_normalized,samples)           #binding blocks filtering for level 0 to filter the possible active candidates
    else:
        df = preprocess.dataPreprocess_rank(df_normalized,samples)             #binding blocks filtering for level 1 to filter the most proactive candidates
    #----------------------------Auto-filtering of outliers with self-adaptive DBSCAN------------------------------
    x_efficiency = np.array(df['S1_Richness_efficiency'])
    x_efficiency = np.reshape(x_efficiency,(len(x_efficiency),1))  
    df_filtered = preprocess.outlierFiltering(x_efficiency,df,1)
    x_balance= np.array(df_filtered['S1_Richness_balance'])
    x_balance = np.reshape(x_balance,(len(x_balance),1))
    df_filtered = preprocess.outlierFiltering(x_balance,df_filtered,2)
    #x_dual = np.array(df_filtered[['Richness_SUM','Richness_STDEV','Richness_COUNT','S1_SUM','S1_STDEV','S1_COUNT']])
    #df_filtered = outlierFiltering(x_dual,df_filtered,3)
    #x_bind_0 = np.array(df_filtered[['S1_ind','Richness_ind','S1_Richness_balance','S1_Richness_efficiency']])
    #df_filtered = outlierFiltering(x_bind_0,df_filtered,1)
    x_bind_1 = np.array(df_filtered[['S1_ind','Richness_ind']])
    df_filtered = preprocess.outlierFiltering(x_bind_1,df_filtered,3,samples)
    #x_bind_2 = np.array(df_filtered[['S1_STDEV','Richness_STDEV']])
    #df_filtered = preprocess.outlierFiltering(x_bind_2,df_filtered,4)
    #----------------------------Add INSR comparison indicator and conditional summation---------------------------
    df_erh_insr = erhAnalysis(erh_dir,df_filtered)
    #----------------------------OCSVM for classification----------------------------------------------------------
    y_predict,x,df_result =classification(df_filtered,opt,samples,currentTime)
    #----------------------------IPCA for Visulization and Hit Candidate Clustering--------------------------------
    df_erh_insr_ipca = similarityAnalysis.ipcaAnalysis(df_erh_insr,df_result)
    #----------------------------Similarity and Ranking------------------------------------------------------------
    df_similarity = similarityAnalysis.similarity(df_erh_insr_ipca,opt)
    #classes=set(y_predict)
    df_score = scoring.score(df_similarity,opt,samples )
    #----------------------------Generate graphs-------------------------------------------------------------------
    fig_1,fig_2,fig_3,fig_4,fig_5,fig_6,fig_7,fig_8,fig_9,fig_10,fig_11= visualize.Visualize(x,df_similarity,y_predict,df_score,samples,opt)

    


    del df_result
    del df_erh_insr
    del df_filtered
    del df_similarity
    del df

    
