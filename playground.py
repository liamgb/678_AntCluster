from scipy.io import arff
import numpy as np
from ant_cluster import AntCluster

data, meta = arff.loadarff("data/iris.arff")
features = np.array([instance.tolist()[:-1] for instance in data])
labels = [instance[-1] for instance in data]

ac = AntCluster(features, label=labels)
ac.similate(10000)

for ant_group in ac.group():
    for ant in ant_group:
        print(ant.gene, ant.label, ant.true_label)