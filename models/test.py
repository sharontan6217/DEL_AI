import sklearn
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,ShuffleSplit,HalvingGridSearchCV
from sklearn.svm import SVC,NuSVC,OneClassSVM,SVR,NuSVR

model_base = OneClassSVM(kernel='rbf',gamma='auto',nu=0.8,coef0=0.1,tol=1e-5)
param_dict = model_base.get_params()
print(param_dict)
param_dict = {  'nu': [0.2,0.4,0.6,0.8],'tol': [1e-3,1e-4,5e-5,1e-5]}
print(param_dict)
model_searchCV = GridSearchCV(model_base,param_dict)
param_dict_searchCV=model_searchCV.get_params()
print(param_dict_searchCV)
#model_searchCV = HalvingGridSearchCV(model_base,param_dict,min_resources=2)

