import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import colors as mcolors
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

#produces heatmap for relating users to one another

data = pd.read_csv("surveyData.csv", encoding = 'utf-8',
                              index_col = ["Timestamp"])
 
# print first 5 rows of zoo data
data= data.drop(data.columns[[5, 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]],axis = 1)

data['How often do you use your smartwatch for the following reasons? [Fitness]'] = data['How often do you use your smartwatch for the following reasons? [Fitness]'].replace(['Never'], 0)
data['How often do you use your smartwatch for the following reasons? [Fitness]'] = data['How often do you use your smartwatch for the following reasons? [Fitness]'].replace(['Infrequently'], 1)
data['How often do you use your smartwatch for the following reasons? [Fitness]'] = data['How often do you use your smartwatch for the following reasons? [Fitness]'].replace(['Sometimes'], 2)
data['How often do you use your smartwatch for the following reasons? [Fitness]'] = data['How often do you use your smartwatch for the following reasons? [Fitness]'].replace(['Often'], 3)
data['How often do you use your smartwatch for the following reasons? [Fitness]'] = data['How often do you use your smartwatch for the following reasons? [Fitness]'].replace(['Always'], 4)



data['How often do you use your smartwatch for the following reasons? [Communication]'] = data['How often do you use your smartwatch for the following reasons? [Communication]'].replace(['Never'], 0)
data['How often do you use your smartwatch for the following reasons? [Communication]'] = data['How often do you use your smartwatch for the following reasons? [Communication]'].replace(['Infrequently'], 1)
data['How often do you use your smartwatch for the following reasons? [Communication]'] = data['How often do you use your smartwatch for the following reasons? [Communication]'].replace(['Sometimes'], 2)
data['How often do you use your smartwatch for the following reasons? [Communication]'] = data['How often do you use your smartwatch for the following reasons? [Communication]'].replace(['Often'], 3)
data['How often do you use your smartwatch for the following reasons? [Communication]'] = data['How often do you use your smartwatch for the following reasons? [Communication]'].replace(['Always'], 4)

data['How often do you use your smartwatch for the following reasons? [Convenience]'] = data['How often do you use your smartwatch for the following reasons? [Convenience]'].replace(['Never'], 0)
data['How often do you use your smartwatch for the following reasons? [Convenience]'] = data['How often do you use your smartwatch for the following reasons? [Convenience]'].replace(['Infrequently'], 1)
data['How often do you use your smartwatch for the following reasons? [Convenience]'] = data['How often do you use your smartwatch for the following reasons? [Convenience]'].replace(['Sometimes'], 2)
data['How often do you use your smartwatch for the following reasons? [Convenience]'] = data['How often do you use your smartwatch for the following reasons? [Convenience]'].replace(['Often'], 3)
data['How often do you use your smartwatch for the following reasons? [Convenience]'] = data['How often do you use your smartwatch for the following reasons? [Convenience]'].replace(['Always'], 4)

data['How often do you use your smartwatch for the following reasons? [Style/ Fashion]'] = data['How often do you use your smartwatch for the following reasons? [Style/ Fashion]'].replace(['Never'], 0)
data['How often do you use your smartwatch for the following reasons? [Style/ Fashion]'] = data['How often do you use your smartwatch for the following reasons? [Style/ Fashion]'].replace(['Infrequently'], 1)
data['How often do you use your smartwatch for the following reasons? [Style/ Fashion]'] = data['How often do you use your smartwatch for the following reasons? [Style/ Fashion]'].replace(['Sometimes'], 2)
data['How often do you use your smartwatch for the following reasons? [Style/ Fashion]'] = data['How often do you use your smartwatch for the following reasons? [Style/ Fashion]'].replace(['Often'], 3)
data['How often do you use your smartwatch for the following reasons? [Style/ Fashion]'] = data['How often do you use your smartwatch for the following reasons? [Style/ Fashion]'].replace(['Always'], 4)

data['How often do you use your smartwatch for the following reasons? [Other]'] = data['How often do you use your smartwatch for the following reasons? [Other]'].replace(['Never'], 0)
data['How often do you use your smartwatch for the following reasons? [Other]'] = data['How often do you use your smartwatch for the following reasons? [Other]'].replace(['Infrequently'], 1)
data['How often do you use your smartwatch for the following reasons? [Other]'] = data['How often do you use your smartwatch for the following reasons? [Other]'].replace(['Sometimes'], 2)
data['How often do you use your smartwatch for the following reasons? [Other]'] = data['How often do you use your smartwatch for the following reasons? [Other]'].replace(['Often'], 3)
data['How often do you use your smartwatch for the following reasons? [Other]'] = data['How often do you use your smartwatch for the following reasons? [Other]'].replace(['Always'], 4)

data = data.rename({'How often do you use your smartwatch for the following reasons? [Fitness]': 'Fitness', 'How often do you use your smartwatch for the following reasons? [Communication]': 'Communication'}, axis=1)  # new method
data = data.rename({'How often do you use your smartwatch for the following reasons? [Convenience]': 'Convenience'}, axis=1)  # new method
data = data.rename({'How often do you use your smartwatch for the following reasons? [Style/ Fashion]': 'Style / Fashion', 'How often do you use your smartwatch for the following reasons? [Other]': 'Other'}, axis=1)  # new method



clusters = 4
 
kmeans = KMeans(n_clusters = clusters)
kmeans.fit(data)
 
print(kmeans.labels_)


pca = PCA(3)
pca.fit(data)

pca_data = pd.DataFrame(pca.transform(data))
 
print(pca_data.head())

colors = list(zip(*sorted((
                    tuple(mcolors.rgb_to_hsv(
                          mcolors.to_rgba(color)[:3])), name)
                     for name, color in dict(
                            mcolors.BASE_COLORS, **mcolors.CSS4_COLORS
                                                      ).items())))[1]
  
  
# number of steps to taken generate n(clusters) colors
skips = math.floor(len(colors[5 : -5])/clusters)
cluster_colors = colors[5 : -5 : skips]

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(pca_data[0], pca_data[1], pca_data[2],
           c = list(map(lambda label : cluster_colors[label],
                                            kmeans.labels_)))
  
str_labels = list(map(lambda label:'% s' % label, kmeans.labels_))
  
list(map(lambda data1, data2, data3, str_label:
        ax.text(data1, data2, data3, s = str_label, size = 16.5,
        zorder = 20, color = 'k'), pca_data[0], pca_data[1],
        pca_data[2], str_labels))
  
plt.show()

# generating correlation heatmap
sns.set(rc={'figure.figsize':(8,8)})
sns.heatmap(data.corr(), annot = True)
 
# posting correlation heatmap to output console
plt.show()
