import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
import random

def kmeansFunc(points, nclusters):
    kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(points)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    df = pd.DataFrame(points, index=labels)
    u_labels = np.unique(labels)
    for label in u_labels:
        plt.scatter(df.loc[label][0], df.loc[label][1])
        
    return (labels, centroids)