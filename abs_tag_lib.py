import re
import numpy as np

from sklearn.metrics import pairwise_distances

# clean the text and use lemmatizer
def clean_text_lemmatize(item,lemmatizer,stopwords):
    
    # remove latex equations
    item = re.sub('\$+.*?\$+','',item)
    
    # tokenize and remove punctuation
    item = re.findall('[a-zA-Z0-9]+',item)
    
    # lowecase everything
    item = [word.lower() for word in item]
    
    # remove english stopwords
    item = [word for word in item if word not in stopwords]
    
    # lemmatize the words
    item = [lemmatizer.lemmatize(word) for word in item]
    
    return item

# Compute the within point scatter for estimating number off clusters to use
def within_point_scatter(X,labels,clusters):
    
    # The classes into which we have clustered the observations
    classes = set(labels)
    
    within_point_scatter = 0

    for l in classes:
        
        # Clustered observations and cluster position
        X_clustered = X[labels == l]
        cluster = clusters[l]
        N_cluster,_ = X_clustered.shape
        
        # Within cluster distance
        within_cluster_distance = pairwise_distances(X_clustered,
                                                     Y=[cluster],
                                                     metric='euclidean',
                                                     n_jobs=-1
                                                    )

        within_point_scatter += N_cluster * np.sum(within_cluster_distance)

    return within_point_scatter

# Analyse the feature importance of words in different clusters
def feature_importance_cluster(clusters,labels,features,max_features = 10):

    for k,cluster in enumerate(clusters):

        # Cluster size
        Nk = np.sum(labels == k)

        # Select relevant features for the cluster and their importance
        bool_relevant = cluster != 0

        relevant_features = features[bool_relevant]
        feature_importance = cluster[bool_relevant]

        cluster_features = sorted(zip(relevant_features,feature_importance),
                                  key=lambda x: x[1],reverse=True
                                 )

        print('Cluster {} - # of items = {}'.format(k,Nk))

        for feature, score in cluster_features[:max_features]:
            print('    - {} : {:.3f}'.format(feature, score))
            
# Create linkage matrix for visualising with scipy dendrogram from sklearn hierarchical model
#
# See https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
#
def linkage_matrix(model):

    n_samples = model.labels_.size

    # size of the cluster at each node in the tree
    cluster_size = []

    for children in model.children_:
        
        size = 0
        
        for child in children:
            if child < n_samples:
                # if it is a leaf node, only one observation is in the cluster
                size += 1
            else:
                # if it is not a leaf node, we can access the size of its children clusters
                size += cluster_size[child - n_samples]
                
        cluster_size.append(size)

    linkage_matrix = np.column_stack([model.children_,
                                      model.distances_,
                                      cluster_size]
                                    )
    
    return linkage_matrix.astype(float)