from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression, ElasticNet, BayesianRidge, HuberRegressor, PoissonRegressor, PassiveAggressiveRegressor
from sklearn.preprocessing import MinMaxScaler,StandardScaler, Normalizer, normalize,minmax_scale
import numpy as np
import pandas as pd
def linearRegression(df_erh):
	scaler = StandardScaler()
	x = df_erh['total_insr']

	for col in df_erh.columns:
		if 'pY1355' in col:
			name = col
			y = df_erh[col]
		elif 'pY1361' in col:
			name = col
			y = df_erh[col]

	x = np.reshape(x,(x.shape[0],1))
	y = np.reshape(y,(y.shape[0],1)) 
	x = scaler.fit_transform(x)
	y = scaler.fit_transform(y)      
	print(x,y)
	model = LinearRegression()
	model.fit(x,y)
	y_predict = model.predict(x)
	lr_score = model.score(x,y)
	df_erh['total_insr_normalized']=x
	df_erh[name+'_normalized']=y
	df_erh['predict']=y_predict
	df_erh['lr_score'] = lr_score
	print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==247) & (df_erh ['CodeA']==169)])
	print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==245) & (df_erh ['CodeA']==191)])
	print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==245) & (df_erh ['CodeA']==235)])
	print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==207) & (df_erh ['CodeA']==134)])
	print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==152) & (df_erh ['CodeA']==173)])
	print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==247) & (df_erh ['CodeA']==174)])
	print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==137) & (df_erh ['CodeA']==233)])
	print(df_erh [(df_erh ['CodeC']==1) & (df_erh ['CodeB']==17) & (df_erh ['CodeA']==71)])
	print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==137) & (df_erh ['CodeA']==233)])
	print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==194) & (df_erh ['CodeA']==202)])
	print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==194) & (df_erh ['CodeA']==52)])
	print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==234) & (df_erh ['CodeA']==173)])        
	print(df_erh [(df_erh ['CodeC']==192) & (df_erh ['CodeB']==245) & (df_erh ['CodeA']==156)])  
	
	return df_erh
