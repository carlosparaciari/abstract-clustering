# Clustering of papers in the *hep-th* category of the ArXiv

### Goal

In this notebook, we want to get a better understanding of the topics researched in the papers submitted to the *hep-th* category on the arXiv (see the [link](https://arxiv.org/list/hep-th/new)). 

To do so, we use the metadata of all papers uploaded on this category during 2015. In particular, we divide the papers into clusters by considering their title and abstract, so that common topics are identified and used to divide the papers.

Additionally, we use one of our classifiers, trained in a different project ([link](https://github.com/carlosparaciari/abstract-classification-embedding)), to label the papers. We then check the relative frequency of the classes in each cluster, to understand whether the natural splits found in the arXiv category are well-captured by the classifier.

### Methods

We use the arXiv dataset of metadata, available on Kaggle at the following [link](https://www.kaggle.com/Cornell-University/arxiv). We specialize our study to those papers in the *hep-th* category (the theoretical side of High Energy Physics), and in particular to those uploaded in 2015 (the most populous recent year for this category).

We use standard NLP tools to clean the dataset (removing punctuation and LaTex equations, removing stopwords, lemmatizing), and use a TF-IDF vectorizer to map each abstract into a high-dimension vector space. Then, we try to visualize this high-dimensional space by using Multi-Dimensional Scaling, a procedure that aims to embed elements of a $d>3$ vector space into a $d=2$ vector space while preserving the relative distances as measured in the high-dimensional space. 

We then cluster the papers using 3 different methods,

- *K-Means*: using the euclidian distance between the vector representation of the papers, we perform K-Means clustering. To identify the optimal number of clusters, we use the within-point scatter and select K to be the one where an elbow in this metric is visible.

- *Hierarchical clustering*: This technique allows us to build clusters from a bottom-up approach, by merging small clusters into bigger ones. The advantage is that we can use this method to visualize how the different clusters merge, and when, in a dendrogram.

- *Spectral clustering*: This approach is most useful when non-convex clusters are present (which might well not be the case here). The idea is to see the similarity matrix as the adjacency matrix of a graph and to use the eigendecomposition of the graph Laplacian to represent each paper (and perform K-means in this representation).

### Result

Each method provides a similar arrangement of clusters. In particular, a number of clusters between 6 and 8 seem to be optimal according to the within-point scatter computed with K-means. The clusters we find in general seems to be connected with the following topics,

- *gauge theory*
- *cosmology*
- *supersymmetry*
- *black hole physics*
- *AdS-CFT correspondence*
- a more general cluster about *quantum field theories* (that is probably a miscellanea of papers).

We use the centres of the clusters to estimate the topic of the papers, by considering the most important features in each centre. This is possible since we use a vector representation such as TF-IDF.

We finally use a Convolutional Neural Network, trained during a different project with a corpus of HEP papers' metadata released by Springer, to classify the papers. We are interested to see if the classes this model creates reflect the natural structure of the arXiv category under investigation. We find that the CNN has indeed been trained on a relevant dataset since it can classify approx 80% of the papers with a probability higher than 1/2 (out of 8 possible classes). Furthermore, when we consider the relative frequency of the classes in each cluster, we see that in the majority of the clusters (4 out of 6), there is a dominant class, reflecting the fact that the structure of the *hep-th* is captured by the classifier we previously trained.

### Sections

- Obtaining the dataset
- Cleaning abstracts and titles
- Pre-processing title and abstracts
- Data visualization
- Clustering
- Predicting and clustering