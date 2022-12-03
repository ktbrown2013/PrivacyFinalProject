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
data= data.drop(data.columns[[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]],axis = 1)

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
data = data.rename({'Please indicate your level of agreement to the following questions.  [I have a good understanding of the information that my smartwatch collects about me.]': 'goodUnderstanding'}, axis=1)  # new method

data['goodUnderstanding'] = data['goodUnderstanding'].replace(['Strongly Disagree'], 0)
data['goodUnderstanding'] = data['goodUnderstanding'].replace(['Disagree'], 1)
data['goodUnderstanding'] = data['goodUnderstanding'].replace(['Neither Agree or Disagree'], 2)
data['goodUnderstanding'] = data['goodUnderstanding'].replace(['Agree'], 3)
data['goodUnderstanding'] = data['goodUnderstanding'].replace(['Strongly Agree'], 4)

'''
topValueI = -1
topValue = ''
for row in data:
    print(row[1] )
    if int(row['Fitness']) > topValueI:
        topValueI = row['Fitness']
        topValue = 'Fitness'
    if int(row['Communication']) > topValueI:
        topValueI = row['Communication']
        topValue = 'Communication'
    if int(row['Convenience']) > topValueI:
        topValueI = row['Convenience']
        topValue = 'Convenience'
    if int(row['Style / Fashion']) > topValueI:
        topValueI = row['Style / Fashion']
        topValue = 'Style / Fashion'
'''
data2 = data.drop(data.columns[5],axis=1)
maxValueIndex = data2.idxmax(axis=1)
print(maxValueIndex)



sns.set_theme(style="whitegrid")

# plotting swarm plot with Age and Height (inches)
fig = plt.figure(figsize=(16,14), dpi=80)
ax = fig.add_subplot(221)
sns.swarmplot(x = data["Fitness"], y = data["goodUnderstanding"], data=data, ax=ax, color="#1d3557")
ax.set_xlim(-1,5)
ax.set_xticks(range(-1,5))
ax.set_yticks(range(-1,5))
ax.set_ylabel("Understanding Level",fontsize=14)
ax.set_xlabel("Fitness",fontsize=14)
ax.set_title("Usage Purpose (Fitness) vs. Understanding Level",fontsize=18)
# set label


ax = fig.add_subplot(222)
sns.swarmplot(x = data["Communication"], y = data["goodUnderstanding"], data=data, ax=ax, color="#2a9d8f")
ax.set_xlim(-1,5)
ax.set_xticks(range(-1,5))
ax.set_yticks(range(-1,5))
ax.set_ylabel("Understanding Level",fontsize=14)
ax.set_xlabel("Communication",fontsize=14)
ax.set_title("Usage Purpose (Communication) vs. Understanding Level",fontsize=18)

ax = fig.add_subplot(223)
sns.swarmplot(x = data["Convenience"], y = data["goodUnderstanding"], data=data, ax=ax, color="#f4a261")
ax.set_xlim(-1,5)
ax.set_xticks(range(-1,5))
ax.set_yticks(range(-1,5))
ax.set_ylabel("Understanding Level",fontsize=14)
ax.set_xlabel("Convenience",fontsize=14)
ax.set_title("Usage Purpose (Convenience) vs. Understanding Level",fontsize=18)

ax = fig.add_subplot(224)
sns.swarmplot(x = data["Style / Fashion"], y = data["goodUnderstanding"], data=data, ax=ax, color="#e76f51")
ax.set_xlim(-1,5)
ax.set_xticks(range(-1,5))
ax.set_yticks(range(-1,5))
ax.set_ylabel("Understanding Level",fontsize=14)
ax.set_xlabel("Style / Fashion",fontsize=14)
ax.set_title("Usage Purpose (Style / Fashion) vs. Understanding Level",fontsize=18)

# display
plt.show()