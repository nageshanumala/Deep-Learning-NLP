# -*- coding: utf-8 -*-
"""
Last amended: 13th March, 2019
Ref:
 1. https://dataplatform.ibm.com/analytics/notebooks/20c1c2d6-6a51-4bdc-9b2c-0e3f2bef7376/view?access_token=52b727bd6515bd687cfd88f929cc7869b0ea420e668b2730c6870e72e029f0d1
 2. http://krasserm.github.io/2018/03/21/bayesian-optimization/

Objectives:
    1. Reading from hard-disk random samples of big-data
    2. Using PCA
    3. Pipelining with StandardScaler, PCA and xgboost
    4. Grid tuning of PCA and xgboost
    5. Randomized search of parameters
    6. Bayes optimization
    7. Feature importance
    8. Genetic algorithm for tuning of parameters
    9. Find feature importance of any Black box estimator
       using eli5 API


"""

################### AA. Call libraries #################
# 1.0 Clear ipython memory
%reset -f

# 1.1 Data manipulation and plotting modules
import numpy as np
import pandas as pd


# 1.2 Data pre-processing
#     z = (x-mean)/stdev
from sklearn.preprocessing import StandardScaler as ss

# 1.3 Dimensionality reduction
from sklearn.decomposition import PCA

# 1.4 Data splitting and model parameter search
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# 1.5 Modeling modules
# conda install -c anaconda py-xgboost
from xgboost.sklearn import XGBClassifier


# 1.6 Model pipelining
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline


# 1.7 Model evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve

# 1.8
import matplotlib.pyplot as plt
from xgboost import plot_importance

# 1.9 Needed for Bayes optimization
from sklearn.model_selection import cross_val_score

# 1.10 Install as: pip install bayesian-optimization
#     Refer: https://github.com/fmfn/BayesianOptimization
from bayes_opt import BayesianOptimization


# 1.11 Find feature importance of ANY BLACK BOX estimator
#      Refer: https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html
#      Install as:
#      conda install -c conda-forge eli5
import eli5
from eli5.sklearn import PermutationImportance


# 1.12 Misc
import time
import os
import gc
import random
from scipy.stats import uniform


# 1.13 Set option to dislay many rows
pd.set_option('display.max_columns', 100)

################# BB. Read data randomly #################
# 2.0 Read random chunks of 10% of data


# 2.1 Set working directory
#os.chdir("C:\\Users\\ashok\\OneDrive\\Documents\\higgsBoson")
#os.chdir("D:\\data\\OneDrive\\Documents\\higgsBoson")
os.chdir("/home/ashok/Documents/10.higgsBoson")
os.listdir()


# 2.2 Count number of lines in the file
#     Data has 250001 rows including header also
tr_f = "training.csv.zip"


# 2.3 Total number of lines and lines to read:
total_lines = 250000
num_lines = 100000


# 2.4 Read randomly 'p' fraction of files
#     Ref: https://stackoverflow.com/a/48589768

p = num_lines/total_lines  # fraction of lines to read

# 2.4.1 Keep the header, then take only p% of lines
#       For each row, if random value from [0,1] interval is
#       greater than 'p' then that row will be skipped

data = pd.read_csv(
         tr_f,
         header=0,
         skiprows=lambda i: i>0 and random.random() > p
         )



# 3.0 Explore data
data.shape                # 100039, 33)
data.columns.values       # Label column is the last one
data.dtypes.value_counts()  # Label column is of object type

# 3.1
data.head(3)
data.describe()
data.Label.value_counts()  # Classes are not unbalanced
                           # Binary data
                           #  b: 65558 , s: 34242

# 3.2 We do not need Id column and Weight column
data.drop(columns = ['EventId','Weight'],inplace = True  )
data.shape                    # (100039, 31); 31 Remining columns


# 3.3 Divide data into predictors and target
#     First 30 columns are predictors
X = data.iloc[ :, 0:30]
X.head(2)

# 3.3.1 30th index or 31st column is target
y = data.iloc[ : , 30]
y.head()


# 3.4 Transform label data to '1' and '0'
#    'map' works element-wise on a Series.
y = y.map({'b':1, 's' : 0})
y.dtype           # int64


# 3.5 Store column names somewhere
#     for use in feature importance

colnames = X.columns.tolist()


# 4. Split dataset into train and validation parts
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.35,
                                                    shuffle = True
                                                    )

# 4.1
X_train.shape        # (65025, 30)
X_test.shape         # (35014, 30)
y_train.shape        # (65025,)
y_test.shape         # (35014,)


################# CC. Create pipeline #################
#### Pipe using XGBoost


# 5 Pipeline steps
# steps: List of (name, transform) tuples
#       (implementing fit/transform) that are
#       chained, in the order in which they
#       are chained, with the last object an
#       estimator.
steps_xg = [('sts', ss() ),
            ('pca', PCA()),
            ('xg',  XGBClassifier(silent = False,
                                  n_jobs=2)        # Specify other parameters here
            )
            ]

# 5.1  Instantiate Pipeline object
pipe_xg = Pipeline(steps_xg)

# 5.2 Another way to create pipeline:
#     Not used below
pipe_xg1 = make_pipeline (ss(),
                          PCA(),
                          XGBClassifier(silent = False,
                                        n_jobs=2)
                          )


##################$$$$$$$$$$$#####################
## Jump now to
##   Either:   Grid Search (DD)
##       Or:   Random Search (EE)
##       Or:   Bayesian Optimization (GG)
##       Or:   Evolutionary Algorithm (HH)
##################$$$$$$$$$$$#####################


##################### DD. Grid Search #################

# 6.  Specify xgboost parameter-range
# 6.1 Dictionary of parameters (16 combinations)
#     Syntax: {
#              'transformerName_parameterName' : [ <listOfValues> ]
#              }
#

parameters = {'xg__learning_rate':  [0.03, 0.05],
              'xg__n_estimators':   [200,  300],
              'xg__max_depth':      [4,6],
              'pca__n_components' : [25,30]
              }                               # Total: 2 * 2 * 2 * 2


# 7  Grid Search (16 * 2) iterations
#    Create Grid Search object first with all necessary
#    specifications. Note that data, X, as yet is not specified
clf = GridSearchCV(pipe_xg,            # pipeline object
                   parameters,         # possible parameters
                   n_jobs = 2,         # USe parallel cpu threads
                   cv =2 ,             # No of folds
                   verbose =2,         # Higher the value, more the verbosity
                   scoring = ['accuracy', 'roc_auc'],  # Metrics for performance
                   refit = 'roc_auc'   # Refitting final model on what parameters?
                                       # Those which maximise auc
                   )

## 7.1 Delete objects not needed
#      We need X_train, y_train, X_test, y_test
del X
del data
del y
gc.collect()

######
#### @@@@@@@@@@@@@@@@@@@ #################
## REBOOT lubuntu MACHINE HERE
#### @@@@ AND NOW WORK IN sublime @@@@@#####


# 7.2. Start fitting data to pipeline
start = time.time()
clf.fit(X_train, y_train)
end = time.time()
(end - start)/60               # 25 minutes



# 7.3
f"Best score: {clf.best_score_} "
f"Best parameter set {clf.best_params_}"

# 7.4. Make predictions
y_pred = clf.predict(X_test)


# 7.5 Accuracy
accuracy = accuracy_score(y_test, y_pred)
f"Accuracy: {accuracy * 100.0}"


# 7.6 Get probbaility of occurrence of each class
y_pred_prob = clf.predict_proba(X_test)

# 7.6.1
y_pred_prob.shape      # (34887, 2)
y_pred_prob


# 7.7 Probability values in y_pred_prob are ordered
#     column-wise, as:
clf.classes_    # array([0, 1]) => Ist col for prob(class = 0)


# 7.8 Draw ROC curve
fpr, tpr, thresholds = roc_curve(y_test,
                                 y_pred_prob[: , 0],
                                 pos_label= 0
                                 )


# 7.9 Plot the ROC curve
fig = plt.figure()          # Create window frame
ax = fig.add_subplot(111)   # Create axes
ax.plot(fpr, tpr)           # Plot on the axes
# 7.9.1 Also connect diagonals
ax.plot([0, 1], [0, 1], ls="--")   # Dashed diagonal line
# 7.9.2 Labels etc
ax.set_xlabel('False Positive Rate')  # Final plot decorations
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve for Higgs Boson particle')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
plt.show()


# 7.10 AUC
auc(fpr,tpr)      # 88.71%


# 7.11
#  Find feature importance of any BLACK Box model
#  Refer: https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html
#  See at the end:  How PermutationImportance works?

# 7.11.1 Instantiate the importance object
perm = PermutationImportance(
                            clf,
                            random_state=1
                            )

# 7.11.2 fit data & learn
#        Takes sometime

start = time.time()
perm.fit(X_test, y_test)
end = time.time()
(end - start)/60


# 7.11.3 Conclude: Get feature weights

"""
# If you are using jupyter notebook, use:

eli5.show_weights(
                  perm,
                  feature_names = colnames      # X_test.columns.tolist()
                  )


"""

fw = eli5.explain_weights_df(
                  perm,
                  feature_names = colnames      # X_test.columns.tolist()
                  )

# 7.11.4 Print importance
fw


##################### EE. Randomized Search #################

# Tune parameters using randomized search
# 8. Hyperparameters to tune and their ranges
parameters = {'xg__learning_rate':  uniform(0, 1),
              'xg__n_estimators':   range(50,300),
              'xg__max_depth':      range(3,10),
              'pca__n_components' : range(20,30)}



# 8.1 Tune parameters using random search
#     Create the object first
rs = RandomizedSearchCV(pipe_xg,
                        param_distributions=parameters,
                        scoring= ['roc_auc', 'accuracy'],
                        n_iter=15,          # Max combination of
                                            # parameter to try. Default = 10
                        verbose = 3,
                        refit = 'roc_auc',
                        n_jobs = 2,          # Use parallel cpu threads
                        cv = 2               # No of folds.
                                             # So n_iter * cv combinations
                        )


# 8.2 Run random search for 25 iterations. 21 minutes
start = time.time()
rs.fit(X_train, y_train)
end = time.time()
(end - start)/60


# 8.3 Evaluate
f"Best score: {rs.best_score_} "
f"Best parameter set: {rs.best_params_} "


# 8.4 Make predictions
y_pred = rs.predict(X_test)


# 8.5 Accuracy
accuracy = accuracy_score(y_test, y_pred)
f"Accuracy: {accuracy * 100.0}"


############### FF. Fitting parameters in our model ##############
###############    Model Importance   #################

# 9. Model with parameters of grid search
model_gs = XGBClassifier(
                    learning_rate = clf.best_params_['xg__learning_rate'],
                    max_depth = clf.best_params_['xg__max_depth'],
                    n_estimators=clf.best_params_['xg__max_depth']
                    )

# 9.1 Model with parameters of random search
model_rs = XGBClassifier(
                    learning_rate = rs.best_params_['xg__learning_rate'],
                    max_depth = rs.best_params_['xg__max_depth'],
                    n_estimators=rs.best_params_['xg__max_depth']
                    )


# 9.2 Modeling with both parameters
start = time.time()
model_gs.fit(X_train, y_train)
model_rs.fit(X_train, y_train)
end = time.time()
(end - start)/60


# 9.3 Predictions with both models
y_pred_gs = model_gs.predict(X_test)
y_pred_rs = model_rs.predict(X_test)


# 9.4 Accuracy from both models
accuracy_gs = accuracy_score(y_test, y_pred_gs)
accuracy_rs = accuracy_score(y_test, y_pred_rs)
accuracy_gs
accuracy_rs



# 10 Get feature importances from both models
%matplotlib qt5
model_gs.feature_importances_
model_rs.feature_importances_
plot_importance(model_gs)
plot_importance(model_rs)
plt.show()


############### GG. Tuning using Bayes Optimization ############
"""
11. Step 1: Define BayesianOptimization function.
            It broadly acts as follows"
            s1. Gets a dictionary of parameters that specifies
                possible range of values for each one of
                the parameters. [Our set: para_set ]
            s2. Picks one value for each one of the parameters
                (from the specified ranges as in (s1)) evaluate,
                a loss-function that is given to it, say,
                accuracy after cross-validation.
                [Our function: xg_eval() ]
            s3. Depending upon the value of accuracy returned
                by the evaluator and also past values of accuracy
                returned, this function, creates gaussian
                processes and picks up another set of parameters
                from the given dictionary of parameters
            s4. The parameter set is then fed back to (s2) above
                for evaluation
            s5. (s2) t0 (s4) are repeated for given number of
                iterations and then final set of parameters
                that optimizes objective is returned

"""
# 11.1 Which parameters to consider and what is each one's range
para_set = {
           'learning_rate':  (0, 1),                 # any value between 0 and 1
           'n_estimators':   (50,300),               # any number between 50 to 300
           'max_depth':      (3,10),                 # any depth between 3 to 10
           'n_components' :  (20,30)                 # any number between 20 to 30
            }


# 11.2 This is the main workhorse
xgBO = BayesianOptimization(
                             xg_eval,     # Function to evaluate performance.
                             para_set     # Parameter set from where parameters will be selected
                             )



# 12 Create a function that when passed some parameters
#    evaluates results using cross-validation

def xg_eval(learning_rate,n_estimators, max_depth,n_components):
    # 12.1 Make pipeline. Pass parameters directly here
    pipe_xg1 = make_pipeline (ss(),                        # Why repeat this here for each evaluation?
                              PCA(n_components=int(round(n_components))),
                              XGBClassifier(
                                           silent = False,
                                           n_jobs=2,
                                           learning_rate=learning_rate,
                                           max_depth=int(round(max_depth)),
                                           n_estimators=int(round(n_estimators))
                                           )
                             )

    # 12.2 Now fit the pipeline and evaluate
    cv_result = cross_val_score(estimator = pipe_xg1,
                                X= X_train,
                                y = y_train,
                                cv = 2,
                                n_jobs = 2,
                                scoring = 'f1'
                                ).mean()             # take the average of all results


    # 12.3 Finally return maximum/average value of result
    return cv_result


# 13. Gaussian process parameters
#     Modulate intelligence of Bayesian Optimization process
gp_params = {"alpha": 1e-5}      # Initialization parameter for gaussian
                                 # process.

# 14. Start optimization. 25minutes
#     Our objective is to maximize results
start = time.time()
xgBO.maximize(init_points=5,    # Number of randomly chosen points to
                                 # sample the target function before
                                 #  fitting the gaussian Process (gp)
                                 #  or gaussian graph
               n_iter=25,        # Total number of times the
               #acq="ucb",       # ucb: upper confidence bound
                                 #   process is to be repeated
                                 # ei: Expected improvement
               # kappa = 1.0     # kappa=1 : prefer exploitation; kappa=10, prefer exploration
              **gp_params
               )
end = time.time()
(end-start)/60


# 15. Get values of parameters that maximise the objective
xgBO.res
xgBO.res['max']




################### HH. Tuning using genetic algorithm ##################
## Using genetic algorithm to find best parameters
#  Ref: https://github.com/rsteca/sklearn-deap
#       https://github.com/rsteca/sklearn-deap/blob/master/test.ipynb

# Install as:
# pip install sklearn-deap
from evolutionary_search import EvolutionaryAlgorithmSearchCV


parameters = {'xg__learning_rate':  [0.03, 0.05],
              'xg__n_estimators':   [200,  300],
              'xg__max_depth':      [4,6],
              'pca__n_components' : [25,30]}


clf2 = EvolutionaryAlgorithmSearchCV(
                                   estimator=pipe_xg,  # How will objective be evaluated
                                   params=parameters,  # Parameters range
                                   scoring="accuracy", # Criteria
                                   cv=2,               # No of folds
                                   verbose=True,
                                   population_size=50,
                                   gene_mutation_prob=0.10,
                                   tournament_size=3,
                                   generations_number=10
                                   )


start = time.time()
clf2.fit(X_train, y_train)   # 1hr 2 minute
end = time.time()
(end-start)/60


clf2.best_params_

# Our cvresults table (note, includes all individuals
#   with their mean, max, min, and std test score).
out = pd.DataFrame(
                  clf2.cv_results_
                   )

out = out.sort_values(
                     "mean_test_score",
                      ascending=False
                      )

out.head()


y_pred_gen = clf2.predict(X_test)
accuracy_gen = accuracy_score(y_test, y_pred_gen)
accuracy_gen    # 81.88 %

####################### I am done ######################

"""
How PermutationImportance works?
Remove a feature only from the test part of the dataset,
and compute score without using this feature. It doesn’t
work as-is, because estimators expect feature to be present.
So instead of removing a feature we can replace it with
random noise - feature column is still there, but it no
longer contains useful information. This method works if
noise is drawn from the same distribution as original
feature values (as otherwise estimator may fail).
The simplest way to get such noise is to shuffle values
for a feature, i.e. use other examples’ feature values -
this is how permutation importance is computed.

"""
