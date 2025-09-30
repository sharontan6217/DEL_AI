from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
def ipca(x,num):

	clf_ipca = IncrementalPCA(n_components=num,batch_size=200)
	clf_ipca  = clf_ipca.fit(x)
	x_ipca = clf_ipca.transform(x)

	return x_ipca