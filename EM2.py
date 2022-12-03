import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from numpy.lib import dstack
from scipy.stats import multivariate_normal
import pandas

#plot distributions
def plot(x: list, y: list, means, variances, K, iterations):
    plt.figure()
    plt.title('Other Users Comfort of Third-Party Data Sharing by Utilization of Smartwatch Features')
    plt.xlabel('Utilization Score(higher=uses more features)')
    plt.ylabel('Comfort Score(higher = less comfortable)')
    plt.plot(x, y, 'go', alpha = 0.5)
    plt.plot(means[:,0], means[:,1], 'ro')
    X, Y = np.meshgrid(linspace(0,25,200), linspace(0,6,200))
    pos = dstack((X,Y))
    for k in range(K):
        Z = multivariate_normal.pdf(pos, means[k,:], variances[k,:,:])
        plt.contour(X,Y,Z, extend = 'min')
    plt.show()

#read in data
data = pandas.read_excel('utilPartyComfortData.xlsx')
data = data.to_numpy()
data = data.T
x = data[:,0]
y = data[:,1]
data = np.vstack((x,y))
print(data)

#set up initial parameters
K = 1
means = np.zeros((K, 2))
variances = np.zeros((K,2,2))
for k in range(K):
    means[k] = np.random.normal(5, .5, size=(1,2))
    variances[k] = np.eye(2)
weights = np.ones((K,1)) / K
#plot(x, y, means, variances, K, 0)
LL = [0]
N = len(x)
r = np.zeros((K,N))

for i in range(30):
    #E-step
    for k in range(K):
        r[k] = weights[k] * multivariate_normal.pdf(data.T, means[k,:], variances[k,:,:])
    r = r / np.sum(r, axis = 0)

    #M-step
    weights = np.sum(r, axis=1) / N
    for k in range(K):
        means[k] = np.sum(r[k] * data, axis=1) / np.sum(r[k])
        diff = data - means[k:k+1].T
        variances[k] = np.dot(r[k] * diff, diff.T) / np.sum(r[k])
    
    if (i%5==0):
        plot(x,y,means,variances,K, i)

    #Loglikelihood
    gaussian = 0
    for k in range(K):
        gaussian += weights[k] * multivariate_normal.pdf(data.T, means[k,:], variances[k,:,:])
    LL.append(-np.sum(np.log(gaussian)))

print(LL)