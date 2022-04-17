
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans
import sklearn 
from sklearn.utils.extmath import row_norms, squared_norm
from scipy.sparse import csc_matrix, csr_matrix
from models.binary_utils import BiConv2dLSR, BiLinearLSR, NoiseTriConv2d, NoiseTriLinear, NoiseBiConv2dLSR
from models.noise_layers import NoiseModule, NoiseConv, NoiseLinear
import torch.nn as nn

import time




def k_means_gpu(weight_vector, n_clusters, verbosity=0, seed=int(time.time()), gpu_id=0):

	if n_clusters == 1:

		mean_sample = np.mean(weight_vector, axis=0)

		weight_vector = np.tile(mean_sample, (weight_vector.shape[0], 1))

		return weight_vector

	elif weight_vector.shape[0] == n_clusters:

		return weight_vector

	elif weight_vector.shape[1] == 1:

		return k_means_cpu(weight_vector, n_clusters, seed=seed)

	else:
		# print('n_clusters', n_clusters)
		# print('weight_vector.shape',weight_vector.shape)
		# print('kmeans++ init start')
		init_centers = sklearn.cluster.k_means_._k_init(X=weight_vector, n_clusters=n_clusters, x_squared_norms=row_norms(weight_vector, squared=True), random_state=None)
		# # print('kmeans++ init finished')
		# # print('init_centers.shape',init_centers.shape)
		centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init=init_centers, yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)

		# centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init="k-means++", yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)

		# centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init="random", yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)
		# centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init="afk-mc2", yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)
		weight_vector_compress = np.zeros((weight_vector.shape[0], weight_vector.shape[1]), dtype=np.float32)
		for v in range(weight_vector.shape[0]):
			weight_vector_compress[v, :] = centers[labels[v], :]
		# weight_compress = np.reshape(weight_vector_compress, (filters_num, filters_channel, filters_size, filters_size))
  
		return weight_vector_compress

def k_means_cpu(weight_vector, space, n_clusters, seed=int(time.time())):

  kmeans_result = KMeans(n_clusters=n_clusters,init=space.reshape(-1,1),n_init=1,algorithm="full").fit(weight_vector)
  print("kmeans clustering")
  #kmeans_result = KMeans(n_clusters=n_clusters, init='k-means++', precompute_distances=True, random_state = seed).fit(weight_vector)
  labels = kmeans_result.labels_
  print("labels",labels.shape)
  centers = kmeans_result.cluster_centers_
  print("centers",centers.shape)
  weight_vector_compress = np.zeros((weight_vector.shape[0], weight_vector.shape[1]), dtype=np.float32)
  for v in range(weight_vector.shape[0]):
      weight_vector_compress[v, :] = centers[labels[v], :]
  # weight_compress = np.reshape(weight_vector_compress, (filters_num, filters_channel, filters_size, filters_size))
  print("weight_vector_compress",weight_vector_compress.shape)
  print 
  return weight_vector_compress

def kmeans_clustering(module,bits=2,propotion=1):

    if hasattr(module, 'linear'):
        dev = module.linear.weight.device
        weight = module.linear.weight.data.cpu().numpy().astype('float32')
        weight_size = torch.numel(module.linear.weight)
    if hasattr(module, 'convs'):
        dev = module.conv.weight.device
        weight = module.conv.weight.data.cpu().numpy().astype('float32') 
        weight_size = torch.numel(module.conv.weight)
    else:
        dev = module.weight.device
        weight = module.weight.data.cpu().numpy().astype('float32')
        weight_size = torch.numel(module.weight)

    #print("weight",np.shape(weight))
    """
    squeezeweight = weight.reshape(weight.shape[0],weight.shape[1])
    shape = squeezeweight.shape
    print("squeezeweight", shape)
    mat = csr_matrix(squeezeweight) if shape[0] < shape[1] else csc_matrix(squeezeweight)
    min_ = min(mat.data)
    max_ = max(mat.data)
    n_clusters = int(weight_size/propotion)

    # weight: cpu tensor
    filters_size = weight.shape[2]
    filters_channel = weight.shape[1]
    filters_num = weight.shape[0]
    space = np.linspace(min_, max_, num= n_clusters)
    
    weight_vector = weight.reshape(-1, filters_size)
    weight_vector_clustered = k_means_cpu(weight_vector=weight_vector.astype('float32'), space=space, n_clusters=n_clusters).astype('float32')

    weight_cube_clustered = weight_vector_clustered.reshape(filters_num, filters_channel, filters_size, -1)
    weight_compress = torch.from_numpy(weight_cube_clustered.astype('float32')).cuda()

    module.weight.data = weight_compress

    if hasattr(module, 'convs'):
        module.conv.weight.data = weight_compress
    if hasattr(module, 'linear'):
        module.linear.weight.data = weight_compress
    else:    
        module.weight.data = weight_compress

    """
    weight = weight.squeeze()
    shape = weight.shape
    if propotion>1:
        sknum = int(weight_size/propotion)
    else:
        sknum = 2**bits

    
    mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
    min_ = min(mat.data)
    max_ = max(mat.data)
    space = np.linspace(min_, max_, num= sknum)
    #init=space.reshape(-1,1),n_init=1,
    #init='k-means++', n_init=10,
    print("mat",len(mat))
    print("bits",bits)

    kmeans = KMeans(n_clusters=len(space),init='k-means++', n_init=10, algorithm="full")
    kmeans.fit(mat.data.reshape(-1,1))
    new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
    #print("new_weight",np.shape(new_weight))
    mat.data = new_weight
    clustered_weight = torch.from_numpy(mat.toarray())
    #print("clustered_weight",np.shape(clustered_weight))
    if isinstance(module, torch.nn.Conv2d):
        clustered_weight = clustered_weight.unsqueeze(-1)
        clustered_weight = clustered_weight.unsqueeze(-1)



    if hasattr(module, 'convs'):
        module.conv.weight.data = clustered_weight.to(dev)
    if hasattr(module, 'linear'):
        module.linear.weight.data = clustered_weight.to(dev)
    else:    
        module.weight.data = clustered_weight.to(dev)
    

    








class KmeansLinear(nn.Module):
    def __init__(self, in_features, out_features, sample_noise=False, noise=0, is_train=True,bits=1):
        super(KmeansLinear, self).__init__()
        self.noise = noise
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.sample_noise = sample_noise
        self.is_train = is_train
        self.bits = bits

    def forward(self, x):
        #print("linearbit",self.bits)
        dev = self.linear.weight.device
        weight = self.linear.weight.data.cpu().numpy()  
        #print("weight",np.shape(weight))
        weight = weight.squeeze()
        #print("weight",np.shape(weight))
        shape = weight.shape
        batch_size = int(weight.shape[0])
        #sknum = int(batch_size/(self.bits))
        sknum = 2**self.bits


        mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
        min_ = min(mat.data)
        max_ = max(mat.data)
        space = np.linspace(min_, max_, num= sknum)
        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, algorithm="full")
        kmeans.fit(mat.data.reshape(-1,1))
        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        #print("new_weight",np.shape(new_weight))
        mat.data = new_weight
        clustered_weight = torch.from_numpy(mat.toarray())
        #print("clustered_weight",np.shape(clustered_weight))
        #clustered_weight = clustered_weight.unsqueeze(-1)
        #clustered_weight = clustered_weight.unsqueeze(-1)
        #print("clustered_weight",np.shape(clustered_weight))
        self.linear.weight.data = clustered_weight.to(dev)
        if not self.noise:
            return self.linear(x)
        # elif self.is_train:
        #     return self.linear(x) + self.noised_foward(x)
        # else:
        #     return self.linear(x) + self.noised_inference(x)
        else:
            return self.linear(x) 

class KmeansConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, sample_noise=False, noise=0, bits=1):
        super(KmeansConv, self).__init__()
        self.noise = noise
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sample_noise = sample_noise
        self.bits = bits

    def forward(self, x):
        #print(self.bits)
        dev = self.conv.weight.device
        weight = self.conv.weight.data.cpu().numpy()  
        #print("weight",np.shape(weight))
        weight = weight.squeeze()
        shape = weight.shape
        batch_size = weight.shape[0]
        #sknum = int(batch_size/(self.bits))
        sknum = 2**self.bits


        mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
        min_ = min(mat.data)
        max_ = max(mat.data)
        space = np.linspace(min_, max_, num= sknum)
        ##init=space.reshape(-1,1), n_init=1,
        kmeans = KMeans(n_clusters=len(space), init='k-means++', n_init=1, algorithm="full")
        kmeans.fit(mat.data.reshape(-1,1))
        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        #print("new_weight",np.shape(new_weight))
        mat.data = new_weight
        clustered_weight = torch.from_numpy(mat.toarray())
        clustered_weight = clustered_weight.unsqueeze(-1)
        clustered_weight = clustered_weight.unsqueeze(-1)
        self.conv.weight.data = clustered_weight.to(dev)

        if not self.noise:
            return self.conv(x)
        else:
            return self.conv(x)+noiseforward(self,x,d=2) 

class KmeansConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, sample_noise=False, noise=0, bits=1,isSemseg=False):
        super(KmeansConv1d, self).__init__()
        self.noise = noise
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sample_noise = sample_noise
        self.bits = bits
        #self.isSemseg = isSemseg

    def forward(self, x):
        sknum = 2**self.bits
        dev = self.conv.weight.device
        weight = self.conv.weight.data.cpu().numpy()

        #print("weight",np.shape(weight))
        weight = weight.squeeze()
        shape = weight.shape
        batch_size = int(weight.shape[0])
        #sknum = int(batch_size/(self.bits))
        
        
      
        mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
        min_ = min(mat.data)
        max_ = max(mat.data)
        space = np.linspace(min_, max_, num= sknum)
        #init='k-means++', n_init=10,
        #init=space.reshape(-1,1), n_init=1,
        kmeans = KMeans(n_clusters=len(space),init='k-means++', n_init=1,  algorithm="full")
        kmeans.fit(mat.data.reshape(-1,1))
        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        #print("new_weight",np.shape(new_weight))
        mat.data = new_weight
        clustered_weight = torch.from_numpy(mat.toarray())
        clustered_weight = clustered_weight.unsqueeze(-1)
        self.conv.weight.data = clustered_weight.to(dev)

        if not self.noise:
            return self.conv(x)
        else:
            return self.conv(x)+noiseforward(self,x,d=1)

