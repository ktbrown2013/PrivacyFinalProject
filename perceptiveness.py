import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import colors as mcolors
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm


data = pd.read_csv("surveyData.csv", encoding = 'utf-8',
                              index_col = ["Timestamp"])
 
# print first 5 rows of zoo data
data= data.drop(data.columns[[0,1,2,3,4,6,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]],axis = 1)

data = data.rename({'Please indicate to the best of your knowledge the answers to the following questions. My smartwatch can track [my location.]': 'Location'}, axis=1)  # new method
data = data.rename({'Please indicate to the best of your knowledge the answers to the following questions. My smartwatch can track [my sleep.]': 'Sleep'}, axis=1)  # new method
data = data.rename({'Please indicate to the best of your knowledge the answers to the following questions. My smartwatch can track [information about my heart.]': 'Heart'}, axis=1)  # new method
data = data.rename({'Please indicate to the best of your knowledge the answers to the following questions. My smartwatch can track [my audio data.]': 'Audio'}, axis=1)  # new method
data = data.rename({'Please indicate to the best of your knowledge the answers to the following questions. My smartwatch can track [and share data with third-parties.]': 'Share'}, axis=1)  # new method
data = data.rename({'Please indicate your level of agreement to the following questions.  [I have a good understanding of the information that my smartwatch collects about me.]': 'Understanding'}, axis=1)  # new method

data['Understanding'] = data['Understanding'].replace(['Strongly Disagree'], 0)
data['Understanding'] = data['Understanding'].replace(['Disagree'], 1)
data['Understanding'] = data['Understanding'].replace(['Neither Agree or Disagree'], 2)
data['Understanding'] = data['Understanding'].replace(['Agree'], 3)
data['Understanding'] = data['Understanding'].replace(['Strongly Agree'], 4)

data['Location'] = data['Location'].replace(['Never'], 0)
data['Location'] = data['Location'].replace(['Infrequently'], 1)
data['Location'] = data['Location'].replace(['Sometimes'], 2)
data['Location'] = data['Location'].replace(['Often'], 3)
data['Location'] = data['Location'].replace(['Always'], 4)

data['Sleep'] = data['Sleep'].replace(['Never'], 0)
data['Sleep'] = data['Sleep'].replace(['Infrequently'], 1)
data['Sleep'] = data['Sleep'].replace(['Sometimes'], 2)
data['Sleep'] = data['Sleep'].replace(['Often'], 3)
data['Sleep'] = data['Sleep'].replace(['Always'], 4)

data['Heart'] = data['Heart'].replace(['Never'], 0)
data['Heart'] = data['Heart'].replace(['Infrequently'], 1)
data['Heart'] = data['Heart'].replace(['Sometimes'], 2)
data['Heart'] = data['Heart'].replace(['Often'], 3)
data['Heart'] = data['Heart'].replace(['Always'], 4)

data['Audio'] = data['Audio'].replace(['Never'], 0)
data['Audio'] = data['Audio'].replace(['Infrequently'], 1)
data['Audio'] = data['Audio'].replace(['Sometimes'], 2)
data['Audio'] = data['Audio'].replace(['Often'], 3)
data['Audio'] = data['Audio'].replace(['Always'], 4)

data['Share'] = data['Share'].replace(['Never'], 0)
data['Share'] = data['Share'].replace(['Infrequently'], 1)
data['Share'] = data['Share'].replace(['Sometimes'], 2)
data['Share'] = data['Share'].replace(['Often'], 3)
data['Share'] = data['Share'].replace(['Always'], 4)

data['Biometric'] = (data['Sleep'] + data['Heart']) / 2
custom_palette = sns.color_palette("Set2", 5)
sns.set_palette(custom_palette)
fig = plt.figure(figsize=(16,16), dpi=80)

ax = fig.add_subplot(221)
sns.boxplot(x = data["Understanding"], y = data["Location"], data=data, ax=ax).set_title("Users Location Tracking Perception by Understanding")
ax.set(xlabel = "Understanding Level", ylabel = "Location Tracking Frequency")

ax = fig.add_subplot(222)
sns.boxplot(x = data["Understanding"], y = data["Biometric"], data=data, ax=ax).set_title("Users Biometric Tracking Perception by Understanding")
ax.set(xlabel = "Understanding Level", ylabel = "Biometric Tracking Frequency")

ax = fig.add_subplot(223)
sns.boxplot(x = data["Understanding"], y = data["Audio"], data=data, ax=ax).set_title("Users Audio Collection Perception by Understanding")
ax.set(xlabel = "Understanding Level", ylabel = "Audio Collection Frequency")

ax = fig.add_subplot(224)
sns.boxplot(x = data["Understanding"], y = data["Share"], data=data, ax=ax).set_title("Users Data Sharing / Aggregation Perception by Understanding")
ax.set(xlabel = "Understanding Level", ylabel = "Data Sharing Frequency")

plt.show()
print(data.head(5))