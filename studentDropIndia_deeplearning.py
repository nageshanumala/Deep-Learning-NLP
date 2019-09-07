"""
Last amended:9th March, 2019
Myfolder: C:\\Users\\ashok\\OneDrive\\Documents\\education_analytics\
          & lubuntu_boost

Objectives:
    a. Education Analytics
    b. Feature engineering
         Calculate using conf rates
         (see file: featureEnf_ConfRates.py
              folder: /home/ashok/Documents/talkingdata)
    b. Feature plotting
    c. Simple missing values imputation
    d. Learning to work with unbalanced education_analytics
    e. Using h2o--deeplearning
    f. What is deeplearning?
    i. Get relative importance of variables
    j. What happens if I do not balance data?--To be done

Ref:
     1.  https://github.com/h2oai/h2o-tutorials/blob/master/tutorials/deeplearning/deeplearning.ipynb
     2.  /home/ashok/Documents/education_analytics/DeepLearningBooklet.pdf

Machine Learning with python and H2O
   https://www.h2o.ai/wp-content/uploads/2018/01/Python-BOOKLET.pdf
H2o deeplearning booklet
   http://docs.h2o.ai/h2o/latest-stable/h2o-docs/booklets/DeepLearningBooklet.pdf
Imbalanced data user guide:
   https://imbalanced-learn.readthedocs.io/en/stable/user_guide.html

"""

# 1.0 Call libraries
%reset -f
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
%matplotlib qt5
# 1.0.1 For measuring time elapsed
from time import time

# 1.1 Working with imbalanced data
# http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.over_sampling.SMOTE.html
# Check imblearn version number as:
#   import imblearn;  imblearn.__version__
from imblearn.over_sampling import SMOTE, ADASYN

# 1.2 Processing data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  OneHotEncoder as ohe
from sklearn.preprocessing import StandardScaler as ss
from sklearn.compose import ColumnTransformer as ct


# 1.3 Data imputation
from sklearn.impute import SimpleImputer

# 1.4 Model building
#     Install h2o as: conda install -c h2oai h2o=3.22.1.2
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

# 1.5 for ROC graphs & metrics
import scikitplot as skplt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
import sklearn.metrics as metrics


# 1.6 Change ipython options to display all data columns
pd.options.display.max_columns = 300



# 2.0 Read data
# os.chdir("C:\\Users\\ashok\\OneDrive\\Documents\\education_analytics\\data")
os.chdir("/home/ashok/Documents/6.education_analytics")
ed = pd.read_csv("studentDropIndia_20161215.csv.zip")

# 2.1 Explore data
ed.head(3)
ed.info()                       # # NULLS in total_toilets, establishment_year


# 2.1.1 Examine distribution of continuous variables
ed.describe()                   # Total toilets is skewed to right
                                # Try:
                                #      ed.boxplot(column = ['total_toilets'])
                                # But do ioilets make any difference to dropouts, try:
                                #     ed.boxplot(column = ['total_toilets'], by = ['continue_drop'])


ed.shape                        # (19100,15)
ed.columns.values               # Target is Ist column: continue_drop
ed.dtypes.value_counts()        # object 5, float64 5, int64  4, bool  1


# 2.2 Summary of target feature
ed['continue_drop'].value_counts()     # continue: 18200, drop: 900


# 2.3 % of data with continue/drop
ed['continue_drop'].value_counts()[1]/ed.shape[0]    # drop: 0.047%

# 2.4 Drop not needed columns
ed.columns
ed.drop(['student_id','school_id'], inplace = True, axis =1)



############# skip this if you are in a hurry for modeling ####################
###########  Data exploration #################################################

# 3.0 Extract subsets of students who are dropping out
#       and those who coninue
df_drop = ed.loc[ed["continue_drop"] == "drop", : ]
df_continue = ed.loc[ed["continue_drop"] == "continue", : ]


## 3.0 Feature plotting and developing relationships

# 3.1 Among the students that are dropping out, which categorical
#     variables or a combination of them impact most
#     The graph also shows that there is a structure among dropouts
fig,ax = plt.subplots(nrows = 2, ncols = 2)
# 3.1.1 Internet vs caste
#       Across caste, dropouts are more wehere Internet = True
#       This needs more investigation or maybe cases where
#       Internet is False are very less
df = pd.crosstab(df_drop.internet, df_drop.caste)
sns.heatmap(df,cmap="YlGnBu", ax = ax[0,0], annot = True)
# 3.1.2 gender vs caste
#       Across caste, females drop more than males
df = pd.crosstab(df_drop.gender, df_drop.caste)
sns.heatmap(df,cmap="YlGnBu", ax = ax[1,0], annot = True)
# 3.1.3 gender vs internet
#       Across gender, drops are more where Internet is True.
#       This needs more investigation or maybe cases where
#       Internet is False are very less
df = pd.crosstab(df_drop.gender, df_drop.internet)
sns.heatmap(df,cmap="YlGnBu", ax = ax[0,1], annot = True)
# 3.1.5 guardian vs caste
#       Across caste, mother's are guardian for most students
#       dropping out
df = pd.crosstab(df_drop.guardian, df_drop.caste)
sns.heatmap(df,cmap="YlGnBu", ax = ax[1,1], annot = True)
plt.show()



# 4. For Total students, examining counts of students across categories
fig = plt.figure()
# 4.1 When gender is female, in most cases mother is dominant guardian
df = pd.crosstab(ed.gender,ed.guardian)
# 4.2  'Mother' followed by 'other' is guardian in most cases
#       Surprisingly 'Father' is 3rd.
ax = fig.add_subplot(221)
sns.heatmap(df,cmap="YlGnBu", annot = True, ax =ax)
# 4.3  Caste vs gender
#      Across all castes population of females are much more
#      than males
df = pd.crosstab(ed.gender,ed.caste)
ax= fig.add_subplot(222)
sns.heatmap(df,cmap="YlGnBu", annot = True,ax= ax)
# 4.4 Caste vs guardian
df = pd.crosstab(ed.guardian,ed.caste)
ax= fig.add_subplot(223)
sns.heatmap(df,cmap="YlGnBu", annot= True, ax =ax)
plt.show()


# 5. How are mean science_marks affected, across categories
#    Heatmap with mean(science_marks) as intensity
fig = plt.figure()
# 5.1 Groupby gender + guardian
grpd = ed.groupby(['gender', 'guardian'])
# 5.2 Summarise mean scainec_marks
df = grpd['science_marks'].mean()
# 5.3 Unstack multiple indicies
df = df.unstack()
# 5.4 Now plot and show
#     Conclusions are clear
sns.heatmap(df)
plt.show()



# 6.0 Relationship of continous variable with the target--Boxplots
f, axes = plt.subplots(2, 2, figsize=(7, 7))
sns.boxplot(x="continue_drop", y="total_students", data=ed, ax=axes[0, 0])
sns.boxplot(x="continue_drop", y="mathematics_marks", data=ed, ax=axes[0, 1])
sns.boxplot(x="continue_drop", y="english_marks", data=ed, ax=axes[1, 0])
sns.boxplot(x="continue_drop", y="science_teacher", data=ed, ax=axes[1, 1])
plt.show()



# 7.0 Relationship of continous variable with the target--Density plots
target_values = ['continue', 'drop']
f, axes = plt.subplots(2, 2, figsize=(7, 7))
for target in target_values:
    # 7. Subset as per target--first 'continue' then 'drop'
    subset = ed[ed['continue_drop'] == target]

    # Draw the density plot
    sns.distplot(subset['total_students'], hist = False, kde = True,
                  label = target, ax = axes[0,0])
    sns.distplot(subset['mathematics_marks'], hist = False, kde = True,
                  label = target, ax = axes[0,1])
    sns.distplot(subset['english_marks'], hist = False, kde = True,
                  label = target, ax = axes[1,0])
    sns.distplot(subset['science_teacher'], hist = False, kde = True,
                  label = target, ax = axes[1,1])

plt.show()



###############################################################################
###########  Data Processing #################################################


# 8.0 Transform boolean values to 1 and 0
#     Inside map(), we can have a lambda function as
#     also dictionary
#     Alternate syntax using lambda:
#     ed['internet'] = ed['internet'].map(lambda x : 1 if (x) else 0 )
ed['internet'] = ed['internet'].map({ True: 1, False:0 })

# 8.1  Which columns have missing data
ed.isnull()
ed.isnull().sum()            # total_toilets: 338, establishment_year: 338


# 8.2  Imputing data. Strategy can be difft for each column
#      For total_toilets apply 'mean'
#      Ref: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
imp = SimpleImputer(strategy="mean")     # Could have been median or constant also
ed['total_toilets'] = imp.fit_transform(ed[['total_toilets']])

# 8.3 For establishment_year, apply mode
imp = SimpleImputer(strategy="most_frequent")
ed['establishment_year'] = imp.fit_transform(ed[['establishment_year']])

# 8.4 Check again
ed.isnull().sum()


# 9.0 Separation into target/predictors
y = ed.iloc[:,0]
X = ed.iloc[:,1:]
X.shape              # 19100  X 12


# 9.1 Which columns are numerical and which categorical?
num_columns = X.select_dtypes(include = ['float64','int64']).columns
num_columns

cat_columns = X.select_dtypes(include = ['object']).columns
cat_columns



# 10. Start creating transformation objects
# 10.1 Tuple for categorical columns
cat = ("cattrans", ohe(), cat_columns)
# 10.2 tuple for numeric columns
num = ("numtrans", ss() , num_columns)
# 10.3 Instantiate column transformer object
colTrans = ct([num,cat])

# 10.4 Fit and transform
X_trans = colTrans.fit_transform(X)
X_trans.shape              # 19100 X 19


## 11.0 Label encoding
#  11.1  Map labels to 1 and 0
y = y.map({"continue" : 1, "drop" : 0})
y.head()



###############################################################################
###########  Data Modeling #################################################


# 12. Split data into train/test
#     train-test split. startify on 'y' variable, Default is None.
X_train, X_test, y_train, y_test =   train_test_split(X_trans,y,test_size = 0.3, stratify = y)

X_train.shape        # 13370 X 19


# 12.1  Process X_train data with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(X_train, y_train)
type(X_res)       # No longer pandas dataframe
                  #  but we will convert to H2o dataframe

# 12.2 Check
X_res.shape                    # 25480 X 19
np.sum(y_res)/len(y_res)       # 0.5 ,earlier ratio was 0.047


# 13.0 Preparing to model data with deeplearning
#      H2o requires composite data with both predictors
#      and target
y_res = y_res.reshape(len(y_res),1)
y_res
X = np.hstack((X_res,y_res))
X.shape            # 25480 X 20


# 13.1 Start h2o
h2o.init()

# 13.2 Transform data to h2o dataframe
df = h2o.H2OFrame(X)
len(df.columns)    # 20
df.shape           # 25480 X 20
df.columns


# 14. Get list of predictor column names and target column names
#     Column names are given by H2O when we converted array to
#     H2o dataframe
X_columns = df.columns[0:19]        # Only column names. No data
X_columns       # C1 to C18
y_columns = df.columns[19]
y_columns

df['C20'].head()      # Just to be sure, Does not show anything in spyder. BUG

# 14.1 For classification, target column must be factor
#      Required by h2o
df['C20'] = df['C20'].asfactor()


# 15. Build a deeplearning model on balanced data
#     http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/deep-learning.html
dl_model = H2ODeepLearningEstimator(epochs=1000,
                                    distribution = 'bernoulli',                 # Response has two levels
                                    missing_values_handling = "MeanImputation", # Not needed by us
                                    variable_importances=True,
                                    nfolds = 2,                           # CV folds
                                    fold_assignment = "Stratified",       # Each fold must be sampled carefully
                                    keep_cross_validation_predictions = True,  # For analysis
                                    balance_classes=False,                # SMOTE is not provided by h2o
                                    standardize = True,                   # z-score standardization
                                    activation = 'RectifierWithDropout',  # Default dropout is 0.5
                                    hidden = [100,100],                  # ## more hidden layers -> more complex interactions
                                    stopping_metric = 'logloss',
                                    loss = 'CrossEntropy')

# 16.1 Train our model
start = time()
dl_model.train(X_columns,
               y_columns,
               training_frame = df)


end = time()
(end - start)/60

# 16.2 Get model summary
print(dl_model)


# 17. Time to make predictions on actual unbalanced 'test' data
#     Create a composite X_test data before transformation to
#     H2o dataframe.
y_test = (y_test.values).reshape(len(y_test), 1)     # Needed to hstack
y_test.shape     # 5730,1
X_test.shape     # 5730,19


# 17.1 Column-wise stack now
X_test = np.hstack((X_test,y_test))         # cbind data
X_test.shape     # 5730,20


# 17,2 Transform X_test to h2o dataframe
X_test = h2o.H2OFrame(X_test)
X_test['C20'] = X_test['C20'].asfactor()


# 18. Make prediction on X_test
result = dl_model.predict(X_test[: , 0:19])
result.shape       # 5730 X 3
result.as_data_frame().head()   # Class-wise predictions


# 18.1 Ground truth
#      Convert H2O frame back to pandas dataframe
xe = X_test['C20'].as_data_frame()
xe['result'] = result[0].as_data_frame()
xe.head()
xe.columns


# 19. So compare ground truth with predicted
out = (xe['result'] == xe['C20'])
np.sum(out)/out.size


# 19.1 Also create confusion matrix using pandas dataframe
f  = confusion_matrix( xe['C20'], xe['result'] )
f


# 19.2 Flatten 'f' now
tp,fp,fn,tn = f.ravel()

# 19.3 Evaluate precision/recall
precision = tp/(tp+fp)
precision                    # 96.29
recall = tp/(tp + fn)
recall                       # 86.95

# 19.4 calculate the fpr and tpr for all thresholds of the classification
pred_probability = result["p1"].as_data_frame()    #  Get probability values and
                                                   #    Convert to pandas dataframe

# 19.5 Get fpr, tpr for various thresholds
fpr, tpr, threshold = metrics.roc_curve(xe['C20'], pred_probability)


# 19.6 Plot AUC curve now
plt.plot(fpr,tpr)
plt.show()


# 19.7 This is the AUC
auc = np.trapz(tpr,fpr)
auc


#  20. Which columns are important
var_df = pd.DataFrame(dl_model.varimp(),
             columns=["Variable", "Relative Importance", "Scaled Importance", "Percentage"])
var_df.head(10)



##############################################################################
##################### Work with no balancing #################################

#  Build deeplearning model using unbalanced data
#  and then predict accuracy on exactly the same 'test's data as above
# 19.0 Preparing to model data with deeplearning
y_train = y_train.reshape(len(y_train),1)
X1 = np.hstack((X_train,y_train))
X1.shape  # 25438 X 20

# 12. Transform data to h2o dataframe
df1 = h2o.H2OFrame(X1)
len(df1.columns)    # 20
df1['C20'].head(3)

# 13. For classification, target column must be factor
df1['C20'] = df1['C20'].asfactor()

# 14. Split train data
# Split the data into Train/Test/Validation with Train having 70% and test and validation 15% each
train1,test1 = df1.split_frame(ratios= [0.7])

# 14.1 Check data sizes
train1.shape
test1.shape


# 15. Build basic deeplearning model on balanced data
dl_model1 = H2ODeepLearningEstimator(epochs=1000,
                                    distribution = 'bernoulli',                 # Response has two levels
                                    missing_values_handling = "MeanImputation", # Not needed by us
                                    variable_importances=True,
                                    nfolds = 2,                           # CV folds
                                    fold_assignment = "Stratified",       # Each fold must be sampled carefully
                                    keep_cross_validation_predictions = True,  # For analysis
                                    balance_classes=False,                # SMOTE is not provided by h2o
                                    standardize = True,                   # z-score standardization
                                    activation = 'RectifierWithDropout',  # Default dropout is 0.5
                                    hidden = [32,32,32],                  # Total layers = hidden + 2
                                    stopping_metric = 'logloss',
                                    loss = 'CrossEntropy')

dl_model1.train(X_columns, y_columns, training_frame = train1)
print(dl_model1)


# 16. Make prediction
result1 = dl_model1.predict(X_test[: , 0:19])
result1[0]
type(result1)    # H2o dataframe

# Compare results
out1 = (result1[0] == X_test['C20'])
type(out1)    # H2o dataframe

# So accuracy score? But accuracy can be deceptive
out1['predict'].sum()/out1.shape[0]

# Truth values are:
y_true = X_test['C20'].as_data_frame()
type(y_true)               # Transformed to pandas dataframe

 # Predicted labels
y_probas = result1[:, 1:].as_data_frame()
y_probas

# Plot ROC graph
skplt.metrics.plot_roc_curve(y_true, y_probas)
plt.show()

# So performance?
tn, fp, fn, tp  = confusion_matrix( X_test['C20'].as_data_frame(),
                                    result1[0].as_data_frame() )


s############################## I am done ###################################
#### AAA
### Try these box plots for feature plotting:
# https://stackoverflow.com/questions/37191983/python-side-by-side-box-plots-on-same-figure?rq=1
# Python Side-by-side box plots on same figure
import numpy
import pandas
from matplotlib import pyplot
import seaborn
seaborn.set(style="ticks")

# Data
df = pandas.DataFrame(numpy.random.rand(10,4), columns=list('ABCD'))
df['E'] = [1, 2, 3, 1, 1, 4, 3, 2, 3, 1]


ax = (
    df.set_index('E', append=True)  # set E as part of the index
      .stack()                      # pull A - D into rows
      .to_frame()                   # convert to a dataframe
      .reset_index()                # make the index into reg. columns
      .rename(columns={'level_2': 'quantity', 0: 'value'})  # rename columns
      .drop('level_0', axis='columns')   # drop junk columns
      .pipe((seaborn.boxplot, 'data'), x='quantity', y='value', hue='E', hue_order=[1, 2])
)
seaborn.despine(trim=True)

################

"""
Undersampling methods:
1. https://imbalanced-learn.readthedocs.io/en/stable/under_sampling.html
2. https://sci2s.ugr.es/keel/pdf/algorithm/congreso/2006-Yen-ICIC.pdf

1. Randomly select majority samples less than the original

2. Create small clusters of majority samples. Instead of randomly
   selecting datapoints from randomly, select these centroids
   as points. Thus all majority data is synthetic or select
   randomly as many points from each cluster as required

3. NearMiss-1: NearMiss-1 selects the majority samples for which
               the average distance to the N closest samples of
               the minority class is the smallest. That is for
               each majority class point, find its average distance
               to say, N=3 nearest minority points. Next, create
               a table of majority-data-point vs avg-distance and
               arrange it in increasing order of avg-distances.
               Select the top M majority data points.

4. NearMiss-2 selects the majority samples for which the average
               distance to the N farthest samples of the minority
               class is the smallest. Once avg-distance is calculated
               follow the same method of selection as for NearMiss-1

5. Applies a nearest-neighbors algorithm and “edit” the dataset by
   removing samples which do not agree “enough” with their neighboorhood

Oversampling methods:

6. ADASYN (from Wikipedia)
    The adaptive synthetic sampling approach, or ADASYN algorithm, builds
    on the methodology of SMOTE, by shifting the importance of the
    classification boundary to those minority classes which are difficult.
    ADASYN uses a weighted distribution for different minority class
    examples according to their level of difficulty in learning, where
    more synthetic data is generated for minority class examples that
    are harder to learn.


"""
