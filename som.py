# Self Organizing Map

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
# We want all rows, and all columns, except one.
X = dataset.iloc[:, :-1].values
# Last column, with all rows. Last column - > credit card application was accepted or not.
y = dataset.iloc[:, -1].values

# Feature Scaling
# Two ways of applying feature scaling - standardization & normalization.
from sklearn.preprocessing import MinMaxScaler

# sc is object of MinMaxScaler class.
# Feature range is between 0 and 1, as we apply normalization, all values of x will be in [0,1]
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
# Minisom is SOM based on numpy, developed. We can write our own too.
from minisom import MiniSom

# We will only apply it on x, as we are doing unsupervised learning, and just want to see features in input.
# X, Y = size of grid.
# input_len -> nos of features in our dataset (X)
# sigma -> radius of different neighbourhoods in grid, default value = 1.0
# learning_rate -> hyperparameter that decided by how much the weights are updated during each iteration.
# The higher the learning rate, the quicker the SOM will take to build, default value = 0.5
# decay_function -> used to improve the convergence in the learning_rate
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)

# Randomly initialize weights to numbers close to 0, but not 0.
# argument = X -> dataset where model is trained.
som.random_weights_init(X)

# Training the SOM -> setps 4 to 9 in README, applied 100 times.
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
# Initialize the window where visualizations will be placed.
bone()

# add all the different colors for the different mean values of the mean inter neuron distances of all winning nodes.
# .T -> transpose of mean intern neuron distance matrix
pcolor(som.distance_map().T)

# gives the legend of all the colors (range of values of the MIDs
# Highest MIDs correspond to the white color, these are the outliers (fraud applications).
colorbar()

# Now we add markers on the map, to see which customers got their 
# application accepted (as fraud cutomers who got accepted are more relevant)
markers = ['o', 's']
colors = ['r', 'g']

# We go through all customers, and check their winning nodes, and depending whether the customer got
# approval or not it will colored red or green
# i -> indices of customers, x -> all vectors of the customers.
for i, x in enumerate(X):
    
    # get winning node for specific customer using winner()
    w = som.winner(x)
    
    # put marker in center of square (winning node)
    # w[0] - x coordinate of wiining node (lower left corner) so add 0.5
    # w[1] - y coordinate of wiining node (lower left corner) so add 0.5
    plot(w[0] + 0.5,
         w[1] + 0.5,
         
         # y has information if customer got approval or not.
         # if customer got approval then y[i] would be 1, and so that corresponds to markers[1] which is 's'
         # if customer gdid not get approval then y[i] would be 0, and so that corresponds to markers[0] which is 'o'
         markers[y[i]],
         
         # edge of marker is colored similarly as above for markers.
         markeredgecolor = colors[y[i]],
         
         # No inside color of marker
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)

print('Fraud Customer IDs')
for i in frauds[:, 0]:
  print(int(i))