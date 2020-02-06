# 1. Write a function to take a list or dictionary of clfs and hypers ie use logistic regression,
# each with 3 different sets of hyper parameters for each
# 2. expand to include larger number of classifiers and hyperparameter settings
# 3. find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal
# clf and parameters settings
# 5. Please set up your code to be run and save the results to the
# directory that its executed from
# 6. Investigate grid search function

# Packages
from warnings import simplefilter
import json
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import csv
# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neural_network import MLPClassifier
print(__doc__)

# Ignore Future Warnings
simplefilter(action = 'ignore', category = FutureWarning)

def GridSearch_Beta(classifiers, data, clf_hyperparameters={}):
    # Store results of K-fold CV into text file
    cv_results = open("./output/cv_results.txt", "w+")
    # Unpack data into separate elements
    X, y, k_folds = data
    # Create Cross-Validation Object
    KFold_CV = KFold(n_splits=k_folds, shuffle=False, random_state=0)
    # Container for Explanation of Results
    cvResultsDict = {}
    clfAccuracyDict = {}
    results_index = 0

    for clfs in classifiers:
        clfString = str(clfs)

        # Iterate through the nested dictionary of hyperparameters
        for clf_key, hyperparameters in clf_hyperparameters.items():
            if clf_key in clfString:
                hyperparameter_key, parameter_values = zip(*hyperparameters.items())
                for values in product(*parameter_values):
                    hyperparameter_set = dict(zip(hyperparameter_key, values))

                    # Iterate through training and test indices by using the split method of the CV object
                    # Split the data into training and target rows for k number of folds
                    for k, (train_index, test_index) in enumerate(KFold_CV.split(X=X, y=y)):
                        clf = clfs(**hyperparameter_set)
                        # the array M subsetted by training indicies trains the X (Training Array)
                        # the array L subsetted by training indicies trains the y (Target Array)
                        clf.fit(X[train_index], y[train_index])
                        # using the training array (X = M), subsetting by the test indicies will
                        # predict the target array (y = L)
                        pred = clf.predict(X[test_index])
                        # add results of the trained classifier at k-fold into results container
                        cvResultsDict[results_index]={'fold': k+1,
                                    'classifier': clf,
                                    'train_index': train_index,
                                    'test_index': test_index,
                                    'accuracy': accuracy_score(y[test_index], pred)}
                        results_index += 1
        # Write results of K-fold CV into text file                
        cv_results.write(str(cvResultsDict))
    # Close K-FOLD CV results to prevent further writes to file
    cv_results.close()

    for key in cvResultsDict:
        k1 = cvResultsDict[key]['classifier']
        v1 = cvResultsDict[key]['accuracy']
        k1Test = str(k1)
        #String formatting
        k1Test = k1Test.replace('            ',' ') # remove large spaces from string
        k1Test = k1Test.replace('          ',' ')

        #Then check if the string value 'k1Test' exists as a key in the dictionary
        if k1Test in clfAccuracyDict:
            clfAccuracyDict[k1Test].append(v1) #append the values to create an array (techically a list) of values
        else:
            clfAccuracyDict[k1Test] = [v1] #create a new key (k1Test) in clfAccuracyDict with a new value, (v1)
    
    # Store results of Classifier Accuracies into csv file
    output_path = './output/'
    try:
       with open(output_path + "clf_acc.csv", 'w', newline="") as csv_file:
           writer = csv.writer(csv_file)
           for key, value in clfAccuracyDict.items():
               writer.writerow([key, value])
    except IOError:
       print("I/O error")
    
    # return cvResultsDict
    # return clfAccuracyDict

    # Plot Accuracy Scores for Each Classifierr in clfAccuracyDict to visualize results 
    # for easy comparison of classifier performance

    n = max(len(v1) for k1, v1 in clfAccuracyDict.items())

    # Plot
    # initialize the plot_num counter for incrementing in the loop below
    plot_num = 1
    filename_prefix = 'clf_Boxplots_'

    # Adjust matplotlib subplots for easy terminal window viewing
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.6      # the top of the subplots of the figure
    wspace = 0.2   # the amount of width reserved for space between subplots,
                # expressed as a fraction of the average axis width
    hspace = 0.2   # the amount of height reserved for space between subplots,
                # expressed as a fraction of the average axis height

    #create the histograms
    #matplotlib is used to create the histograms: https://matplotlib.org/index.html
    for k1, v1 in clfAccuracyDict.items():
        # for each key in our clfAccuracyDict, create a new histogram with a given key's values
        fig = plt.figure(figsize =(10,10)) # This dictates the size of our boxplots
        ax  = fig.add_subplot(1, 1, 1) # As the ax subplot numbers increase here, the plot gets smaller
        plt.boxplot(v1, vert = False) # create the boxplots with the values
        ax.set_title(k1, fontsize=25) # increase title fontsize for readability
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
        plt.savefig(output_path + 'plots/'+ filename, bbox_inches = 'tight') # save the plot to the user's working directory
        plot_num = plot_num+1 # increment the plot_num counter by 1
    plt.show()

    
    


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
results = GridSearch_Beta(classifiers, data, clf_hyperparameters)

# Investigate GridSearch: https://scikit-learn.org/stable/modules/grid_search.html
# GridSearch from Python's scikit-learn package attempt to search and recommend the optimal hyper-parameter space
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
 