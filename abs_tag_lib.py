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

# Maps the words into the text to the corresponding index in the word2vec model
#
# NOTE: Since we later pad with 0's, and the index 0 is associated with a particular
#       word in our w2v model, we shift the index of every word by 1.
#
def hashing_trick(text,w2v):
    return np.array([w2v.vocab[word].index+1 for word in text if word in w2v.vocab])

# Maps each sentence in dataframe into a padded sequence of integer (pad is done with 0)
def hashed_padded_sequences(df_text,w2v):
    
    hashed_text = df_text.apply(hashing_trick,args=(w2v,))
    
    max_length = hashed_text.apply(len).max()
    hashed_padded_text = hashed_text.apply(lambda arr : np.pad(arr,(0,max_length-arr.size)))
    
    return hashed_padded_text

# Assign a class based on the probability vector produced by the model
#
# NOTE : This functions assumes that the last of the labels is an "unknown label",
# which is used when the model is uncertain about the class of the paper.
#
def predict_with_treshold(X, labels, model, treshold=0.5):

    N,_ = X.shape

    class_probability = model.predict(X)
    item_to_class = dict(np.argwhere(class_probability > treshold))
    classification = [item_to_class.get(index,-1) for index in range(N)]

    return labels[classification]