This is for "A Hybrid Unsupervised Methodology on Artificial Intelligence Filtering for automatically processing cellular DNA-Encoded Library (DEL) Datasets".

Data:
Please contact us for dataset: xtan@cs.hku.hk, or try your own dataset.

Installation:
Please run 'pip -r install requirements.txt'

Experiment:
The complete command is:
Example:
```shell
python experiment.py 
    --data_dir {data_dir} \
    --erh_dir {erh_dir} \
    --graph_dir {graph_dir} \
    --graph_dir {graph_dir} \
    --output {output_dir} \
    --model_name {clustering_model_name} \
    --parameter_training {parameter_training} \
    --amplify_filtering {amplify_filtering}
```

- `data_dir`: type=str,default='./data/phase_1/total.csv', 'directory of the original data' 
- `erh_dir`,type=str,default='./data/phase_1/erh/', 'directory of erh files' 
- `graph_dir`,type=str,default='./graph/phase_1/level_1/', 'directory of graphs'
- `output_dir`,type=str,default='./output/phase_1/level_1/', 'directory of outputs'
- `level`,type=str,default='level1', 'level is either "level0" or "level1". Level0 is for filtering, and level1 is for ranking.'
- `model_name`,type=str,default='OneClassSVM', 'clustering model is one of the list ["OneClassSVM","KMeans","Spectral","BIRCH","AgglomerativeClustering","OpticsClustering"].'
- `parameter_optimizer`,type=str,default='Yes', 'OCSVM model can be auto-optimized.Default to "Yes". Can set to "No" to use the default model.'
- `amplify_filtering`,type=str,default='No', 'We add a few amplifications for large dataset in preporcessing as an option.  Input "Yes" if the dataset is large.'

To filter 30 million dataset
please run:
```shell
'python experiment.py --level level0'
```
or
```shell
'python experiment.py --level level1'
```
To filter 1.03 billion dataset, we tune a few parameters to amplify the filtering in preprocess stage. You can tune more by yourself for your datasets. 
please run:
```shell
'python experiment.py --level level0 --amplify_filtering Yes'
```
or
```shell
'python experiment.py --level level1 --amplify_filtering Yes'
```