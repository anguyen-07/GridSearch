#'''An exhaustive GridSearch function with a list supervised learning classifiers (i.e. logistic regression, random forest), some data, and a dictionary of hyperparameters as input. The function returns the optimized classifier along with the hyperparameter value combinations that produces the highest accuracy score or chosen performance metric. Plots are generated with matplotlib to visualize the performance of competing models and will assist in identifying the optimal classification model.'''

# Packages
from warnings import simplefilter
import json
import statistics
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import csv
import datetime
import os
# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neural_network import MLPClassifier
print(__doc__)

# Ignore Future Warnings
simplefilter(action = 'ignore', category = FutureWarning)

# Run Cross-Validation with list of classifiers and inputted list of hyperparameters
def run_CV(classifier, data, clf_hyperparameters ={}):
    # Unpack data into separate elements
    X, y, k_folds = data
    # Create Cross-Validation Object
    KFold_CV = KFold(n_splits=k_folds, shuffle=False, random_state=0)
    # Container for Explanation of Results
    cvResultsDict = {}
    
    # Split the data into training and target rows for k number of folds
    for k, (train_index, test_index) in enumerate(KFold_CV.split(X=X, y=y)):
        # Unpack hyperparameters into classifier
        clf = classifier(**clf_hyperparameters)
        clf.fit(X[train_index], y[train_index])
        # using the training array (X), subsetting by the test indicies wil lpredict the target array (y)
        pred = clf.predict(X[test_index])
        # store accuracies and metadata of the trained classifier at k-fold into results container
        cvResultsDict[k] = {'fold': k + 1,
                    'classifier': clf,
                    'train_index': train_index,
                    'test_index': test_index,
                    'accuracy': accuracy_score(y[test_index], pred)}
    return cvResultsDict


# Create dictionary to store accuracy scores by classifier and unique hyperparameter combinations
clfAccuracyDict = {}
def populateClfAccuracyDict(dict):
    '''Populates classification accuracy dictionary after fitting each specified model'''

    for key in dict:
        classifier = dict[key]['classifier']
        accuracy = dict[key]['accuracy']
        # Since each classifier has a pre-defined number of k-folds, check if the string value of the key exists in the dictionary to prevent multiple 'key' values with the same classifier and hyperparameter settings
        clfTest = str(classifier)
        # Format key value string to remove large spaces
        clfTest = clfTest.replace('            ',' ')
        clfTest = clfTest.replace('          ',' ')
        # If it exists as a key, append accuracy score values as a list/array under that key. If it does not exist as a key, create a new key in the dictionary.
        if clfTest in clfAccuracyDict:
            clfAccuracyDict[clfTest].append(accuracy)
        else:
            clfAccuracyDict[clfTest] = [accuracy]


def best_score(GridSearchResults):
    '''Saves the best score from the executed gridsearch.'''

    # Determine best median accuracy score for models across number of k-folds
    best_accuracy = max(statistics.mean(accuracy) for classifier, accuracy in GridSearchResults.items())
    optimal_hyperparameters = ()
    best_accuracy = 0
    filename = 'best_clf_Boxplot'

    for classifier, accuracy in GridSearchResults.items():
        if best_accuracy < statistics.mean(accuracy):
            best_accuracy = statistics.mean(accuracy)
            optimal_hyperparameters = classifier, best_accuracy

            fig = plt.figure(figsize=(20,10)) # This dictates the size of the plot
            ax = fig.add_subplot(1,1,1) # As the ax subplot numbers increase here, the plot gets smaller
            plt.boxplot(accuracy, vert = False) # create the plot with the desired values
            ax.set_title(str(classifier) + "\nAccuracy: " + str(best_accuracy), fontsize = 30) # increase title fontsize for readability
            ax.set_xlabel('Accuracy Scores', fontsize=25) # increase x-axis label fontsize for readability
            ax.set_ylabel('Classifer Accuracy (By K-Fold)', fontsize=25) # increase y-axis label fontsize for readability 
            ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1)) # The accuracy can only be from 0 to 1 (e.g. 0 or 100%)
            ax.yaxis.set_ticks(np.arange(0, 1, 1)) # n represents the number of k-folds
            ax.xaxis.set_tick_params(labelsize=20) # increase x-axis tick fontsize for readability
            ax.yaxis.set_tick_params(labelsize=20) # increase y-axis tick fontsize for readability
            # ax.grid(True) # you can turn this on for a grid, but I think it looks messy here.
            # plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
            plt.savefig(output_dir_best + '/' + filename, bbox_inches = 'tight') # save plot to working directory

    # Display results of best accuracy score for models across classifiers in the GridSearch
    # print("Check current working directory for results and plots.")
    # print("Best Accuracy: ", best_accuracy)
    # print(optimal_hyperparameters)
    print("\n*Model with best accuracy: ", optimal_hyperparameters[1], "\nClassifier & Parameters: \n", optimal_hyperparameters[0], "\n*")


def GridSearch(classifiers, data, clf_hyperparameters={}):
    '''Main GridSearch function that takes list of classification models and dictionary of hyperparameters as input.'''

    # Container to store accuracy scores by classifier and unique hyperparameter combinations (used in populateClfAccuracyDict function)

    for clf in classifiers:
        clfString = str(clf)

        # Iterate through the nested dictionary of hyperparameters
        for clf_key, hyperparameters in clf_hyperparameters.items():
            if clf_key in clfString: # Check if values in list of classifiers are in the dictionary of classifier hyperparameters
                 hyperparameter_key, parameter_values = zip(*hyperparameters.items()) # Map hyperparameter key with the corresponding index of its hyperparameter value within the inner nested dictionary
                 for values in product(*parameter_values): # Extract the unique combinations of the hyperparameter values in inner dictionray with a Cartesian product [itertools.product()]
                    hyperparameter_set = dict(zip(hyperparameter_key, values)) # Store unique combination of hyperparmeters in a dictionary
                    cvResults = run_CV(clf, data, hyperparameter_set) # Use the classifier and hyperparameter dictionary to run a cross-validation
                    populateClfAccuracyDict(cvResults) # store results of CV in the classifier accuracy dictionary: clfAccuracyDict
    
    # Store results of Classifier Accuracies into csv file
    try:
       with open(output_dir + "clf_acc.csv", 'w', newline="") as csv_file:
           writer = csv.writer(csv_file)
           for key, value in clfAccuracyDict.items():
               writer.writerow([key, value])
    except IOError:
       print("I/O error")

    best_score(clfAccuracyDict)
    
# Visualize accuracy score of competing classifiers
def plot_GridSearch(GridSearchResults):
    '''Plot the gridsearch results of competing models to visualize results for easy comparison of classifier performance.'''

    # Determine maximum frequency (# k-folds) for histogram y-axis 
    # n = max(len(v1)) for k1, v1 in clfAccuracyDict.items()
    # Initialize plot number counter for incrementing in the for loop below & plot name
    filename_prefix = 'clf_Boxplots_'
    plot_num = 1

    # Adjust matplotlib subplots for easy terminal window viewing
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.6      # the top of the subplots of the figure
    wspace = 0.2   # the amount of width reserved for space between subplots, expressed as a fraction of the average axis width
    hspace = 0.2   # the amount of height reserved for space between subplots,expressed as a fraction of the average axis height

    # Create the visualization - matplotlib is used to create the plots: https://matplotlib.org/index.html
    for classifier, accuracy in GridSearchResults.items():
        # for each key in our clfAccuracyDict, create a new histogram with a given key's values
        fig = plt.figure(figsize =(10,10)) # This dictates the size of our plots
        ax  = fig.add_subplot(1, 1, 1) # As the ax subplot numbers increase here, the plot gets smaller
        plt.boxplot(accuracy, vert = False) # create the plot with the desired values
        ax.set_title(classifier, fontsize=25) # increase title fontsize for readability
        ax.set_xlabel('Accuracy Scores', fontsize=25) # increase x-axis label fontsize for readability
        ax.set_ylabel('Classifer Accuracy (By K-Fold)', fontsize=25) # increase y-axis label fontsize for readability
        ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1)) # The accuracy can only be from 0 to 1 (e.g. 0 or 100%)
        ax.yaxis.set_ticks(np.arange(0, 1, 1)) # n represents the number of k-folds
        ax.xaxis.set_tick_params(labelsize=20) # increase x-axis tick fontsize for readability
        ax.yaxis.set_tick_params(labelsize=20) # increase y-axis tick fontsize for readability
        #ax.grid(True) # you can turn this on for a grid, but I think it looks messy here.

        # pass in subplot adjustments from above.
        plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
        plot_num_str = str(plot_num) #convert plot number to string
        filename = filename_prefix + plot_num_str # concatenate the filename prefix and the plot_num_str
        plt.savefig(output_dir_GridSearchResults + '/' + filename, bbox_inches = 'tight') # save the plot to the user's working directory
        plot_num = plot_num+1 # increment the plot_num counter by 1
    plt.show()


# Current data and time to format output folder structure
output_dir = 'result' + '/'
output_dir_best = output_dir + '/' + 'BestResult'
output_dir_GridSearchResults = output_dir + '/' + 'CompleteResults'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(output_dir_best):
    os.makedirs(output_dir_best)
if not os.path.exists(output_dir_GridSearchResults):
    os.makedirs(output_dir_GridSearchResults)

# Load Data
from sklearn import datasets
bCancer = datasets.load_breast_cancer()
X = bCancer.data # X: Training Data/Matrix (check: X.shape)
y = bCancer.target # y: Target Data/Labels Matrix (check y.shape[0])

# EDIT: a single value, 5, to use for 5-fold (k-fold) cross validation
k_folds = 5
# EDIT: pack the arrays together into "data"
data=(X, y, k_folds)

# Specify Classifiers
classifiers = [RandomForestClassifier, LogisticRegression, KNeighborsClassifier]

# Specify Hyperparmaters to be grid searched as a dictionary
# Specify range of hyperparameter values to be tested as a nested dictionary
clf_hyperparameters={'RandomForestClassifier': {"n_estimators": [10, 100, 1000, 5000],
																 "max_features": ['auto', 'sqrt', 'log2'],
																 "max_depth":[None, 10, 25, 50],
																 "min_samples_split":[0.5,0.8,2,5,10],
																 "min_samples_leaf":[0.1,0.5,1,5,10]},
                     'LogisticRegression': {"solver": ['newton-cg', 'sag', 'lbfgs'],
                                            "tol":[0.0001, 0.001, 0.01, 0.1],
                                            "max_iter":[50,100,150,300],
                                            "class_weight": [None, 'balanced']},
                     'KNeighborsClassifier': {"n_neighbors": [3, 4, 5, 7],
                       						  "algorithm":['ball_tree','kd_tree','brute','auto'],
                       						  "p": [1, 2, 3],
                       						  "leaf_size":[5,10,30]}}

# Test
GridSearch(classifiers, data, clf_hyperparameters)
plot_GridSearch(clfAccuracyDict)

# GridSearch algorithms from scikit-learn: https://scikit-learn.org/stable/modules/grid_search.html
# GridSearch from Python's scikit-learn package attempts to search and recommend the optimal hyper-parameter space
# that produces the best cross-validation score for a given estimator (regressor/classifier). In this example of 
# our user-defined function, we used K-fold CV in an Exhaustive Grid Search method with the grid of hyperparameter
# values specified in the clf_hyperparameters dictionary. In this method, all possible combinations of values in the
# hyperparameter are passed into the given estimator/model to return a cross-validation score that helps determine 
# which hyperparameter values are optimal given the specific data. 
#
# from sklearn.model_selection import GridSearchCV
# The function call for an Exhaustive Grid Search from the scitkit-learn package is GridSearchCV(). The documentation 
# can be found here: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html. The
# documentation provides a quick description of the algorithm and the passable parameters to define the search.
# 
# Another Grid Search method is the Randomized Parameter Optimization that uses a randomized search over parameters,
# given that the parameters are sampled from a distribution over possible parameter values. The major benefit of this 
# over an exhaustive search is that the processing time can be greatly decreased, but the tradeoff is that there is no
# guarantee of finding the optimal combination of hyperparameters.
#
# from sklearn.model_selection import RandomizedSearchCV
# The function call for a Randomized Grid Search from the scitkit-learn package is RandomizedSearchCV(). The documentation 
# can be found here: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html. 
# The documentation provides a quick description of the algorithm and the passable parameters to define the search.
#
# All grid searches consist of an estimator (regressor or classifier such as RandomForest), a hyperparameter space, a
# method for searching or sampling hyperparameter values (Exhaustive vs. Random), a cross-validation scheme, and a score
# function (i.e. accuracy score, r2 score, etc).