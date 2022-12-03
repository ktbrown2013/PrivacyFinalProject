import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import colors as mcolors
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import numpy as np
import matplotlib.patches as mpatches

data = pd.read_csv("surveyData.csv", encoding = 'utf-8',
                              index_col = ["Timestamp"])
 
# print first 5 rows of zoo data
data= data.drop(data.columns[[0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]],axis = 1)

data = data.rename({'Please indicate your level of agreement to the following questions.  [I have a good understanding of the information that my smartwatch collects about me.]': 'Understanding'}, axis=1)  # new method
data = data.rename({'Have you personally made manual changes to your smartwatchs data privacy settings?': 'Manual'}, axis=1)  # new method
data = data.rename({'Do you have concerns about privacy issues regarding your device? ': 'Concern'}, axis=1)  # new method
data = data.rename({'Do you trust your smartwatch manufacturer with your personal data?': 'Manufacturer'}, axis=1)  # new method
data = data.rename({'Would you now modify your current privacy settings or change the way you use your smartwatch?': 'Change'}, axis=1)  # new method

data['Understanding'] = data['Understanding'].replace(['Strongly Disagree'], 0)
data['Understanding'] = data['Understanding'].replace(['Disagree'], 1)
data['Understanding'] = data['Understanding'].replace(['Neither Agree or Disagree'], 2)
data['Understanding'] = data['Understanding'].replace(['Agree'], 3)
data['Understanding'] = data['Understanding'].replace(['Strongly Agree'], 4)

data['Manual'] = data['Manual'].replace(['Yes'], 1)
data['Manual'] = data['Manual'].replace(['No'], 0)

data['Concern'] = data['Concern'].replace(['Yes'], 1)
data['Concern'] = data['Concern'].replace(['No'], 0)

data['Manufacturer'] = data['Manufacturer'].replace(['Yes'], 1)
data['Manufacturer'] = data['Manufacturer'].replace(['No'], 0)

data['Change'] = data['Change'].replace(['Yes'], 1)
data['Change'] = data['Change'].replace(['No'], 0)

total0 = (len(data.query('Understanding == 0 and Change == 1')+data.query('Understanding == 0 and Change == 0')))
change0 = (len(data.query('Understanding == 0 and Change == 1')))
concern0 = (len(data.query('Understanding == 0 and Change == 1 and Concern == 1')))
manual0 = (len(data.query('Understanding == 0 and Change == 1 and Manual == 1')))
manufacturer0 = (len(data.query('Understanding == 0 and Change == 1 and Manufacturer == 1')))

total1= (len(data.query('Understanding == 1 and Change == 1')+data.query('Understanding == 1 and Change == 0')))
change1=(len(data.query('Understanding == 1 and Change == 1')))
concern1=(len(data.query('Understanding == 1 and Change == 1 and Concern == 1')))
manual1=(len(data.query('Understanding == 1 and Change == 1 and Manual == 1')))
manufacturer1=(len(data.query('Understanding == 1 and Change == 1 and Manufacturer == 1')))

total2=(len(data.query('Understanding == 2 and Change == 1')+data.query('Understanding == 2 and Change == 0')))
change2=(len(data.query('Understanding == 2 and Change == 1')))
concern2=(len(data.query('Understanding == 2 and Change == 1 and Concern == 1')))
manual2=(len(data.query('Understanding == 2 and Change == 1 and Manual == 1')))
manufacturer2=(len(data.query('Understanding == 2 and Change == 1 and Manufacturer == 1')))

total3=(len(data.query('Understanding == 3 and Change == 1')+data.query('Understanding == 3 and Change == 0')))
change3=(len(data.query('Understanding == 3 and Change == 1')))
concern3=(len(data.query('Understanding == 3 and Change == 1 and Concern == 1')))
manual3=(len(data.query('Understanding == 3 and Change == 1 and Manual == 1')))
manufacturer3=len(data.query('Understanding == 3 and Change == 1 and Manufacturer == 1'))

total4=(len(data.query('Understanding == 4 and Change == 1')+data.query('Understanding == 4 and Change == 0')))
change4=(len(data.query('Understanding == 4 and Change == 1')))
concern4=(len(data.query('Understanding == 4 and Change == 1 and Concern == 1')))
manual4=(len(data.query('Understanding == 4 and Change == 1 and Manual == 1')))
manufacturer4=(len(data.query('Understanding == 4 and Change == 1 and Manufacturer == 1')))

df = pd.DataFrame([['0',change0,concern0,manual0,manufacturer0],['1',change1,concern1,manual1,manufacturer1],['2',change2,concern2,manual2,manufacturer2],['3',change3,concern3,manual3,manufacturer3], 
['4',concern4,change4,manual4,manufacturer4]],columns=['Initial Understanding','Will Change','Is Concerned','Has Made Changes','Trusts Manufacturer'])

plt.style.use('seaborn-dark-palette')
fig = plt.figure(figsize=(16,16), dpi=80)
ax=fig.add_subplot(211)
df.plot(x='Initial Understanding', ylabel= "Users",
        kind='bar',
        stacked=False,
        title='Behaviour Analysis of Users Intending to Change Privacy Settings',colormap='Set3')

total0 = (len(data.query('Understanding == 0 and Change == 1')+data.query('Understanding == 0 and Change == 0')))
change0 = (len(data.query('Understanding == 0 and Change == 0')))
concern0 = (len(data.query('Understanding == 0 and Change == 0 and Concern == 1')))
manual0 = (len(data.query('Understanding == 0 and Change == 0 and Manual == 1')))
manufacturer0 = (len(data.query('Understanding == 0 and Change == 0 and Manufacturer == 1')))

total1= (len(data.query('Understanding == 1 and Change == 1')+data.query('Understanding == 1 and Change == 0')))
change1=(len(data.query('Understanding == 1 and Change == 0')))
concern1=(len(data.query('Understanding == 1 and Change == 0 and Concern == 1')))
manual1=(len(data.query('Understanding == 1 and Change == 0 and Manual == 1')))
manufacturer1=(len(data.query('Understanding == 1 and Change == 0 and Manufacturer == 1')))

total2=(len(data.query('Understanding == 2 and Change == 1')+data.query('Understanding == 2 and Change == 0')))
change2=(len(data.query('Understanding == 2 and Change == 0')))
concern2=(len(data.query('Understanding == 2 and Change == 0 and Concern == 1')))
manual2=(len(data.query('Understanding == 2 and Change == 0 and Manual == 1')))
manufacturer2=(len(data.query('Understanding == 2 and Change == 0 and Manufacturer == 1')))

total3=(len(data.query('Understanding == 3 and Change == 1')+data.query('Understanding == 3 and Change == 0')))
change3=(len(data.query('Understanding == 3 and Change == 0')))
concern3=(len(data.query('Understanding == 3 and Change == 0 and Concern == 1')))
manual3=(len(data.query('Understanding == 3 and Change == 0 and Manual == 1')))
manufacturer3=len(data.query('Understanding == 3 and Change == 0 and Manufacturer == 1'))

total4=(len(data.query('Understanding == 4 and Change == 1')+data.query('Understanding == 4 and Change == 0')))
change4=(len(data.query('Understanding == 4 and Change == 0')))
concern4=(len(data.query('Understanding == 4 and Change == 0 and Concern == 1')))
manual4=(len(data.query('Understanding == 4 and Change == 0 and Manual == 1')))
manufacturer4=(len(data.query('Understanding == 4 and Change == 0 and Manufacturer == 1')))

df2 = pd.DataFrame([['0',change0,concern0,manual0,manufacturer0],['1',change1,concern1,manual1,manufacturer1],['2',change2,concern2,manual2,manufacturer2],['3',change3,concern3,manual3,manufacturer3], 
['4',concern4,change4,manual4,manufacturer4]],columns=['Initial Understanding','Will Not Change','Is Concerned','Has Made Changes','Trusts Manufacturer'])

fig.add_subplot(212)
ax = df2.plot(x='Initial Understanding', ylabel= "Users",
        kind='bar',
        stacked=False,
        title='Behaviour Analysis of Users Not Intending to Change Privacy Settings',colormap='Set3')

plt.show()