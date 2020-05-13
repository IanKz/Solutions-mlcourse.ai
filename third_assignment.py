# Solution for third's week assignment

from matplotlib import pyplot as plt
import math
plt.rcParams['figure.figsize'] = (10, 8)
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import collections
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from io import StringIO
import pydotplus 

# Create dataframe with dummy variables
def create_df(dic, feature_list):
    out = pd.DataFrame(dic)
    out = pd.concat([out, pd.get_dummies(out[feature_list])], axis = 1)
    out.drop(feature_list, axis = 1, inplace = True)
    return out

# Some feature values are present in train and absent in test and vice-versa.
def intersect_features(train, test):
    common_feat = list( set(train.keys()) & set(test.keys()))
    return train[common_feat], test[common_feat]

features = ['Looks', 'Alcoholic_beverage','Eloquence','Money_spent']

df_train = {}
df_train['Looks'] = ['handsome', 'handsome', 'handsome', 'repulsive',
                         'repulsive', 'repulsive', 'handsome'] 
df_train['Alcoholic_beverage'] = ['yes', 'yes', 'no', 'no', 'yes', 'yes', 'yes']
df_train['Eloquence'] = ['high', 'low', 'average', 'average', 'low',
                                   'high', 'average']
df_train['Money_spent'] = ['lots', 'little', 'lots', 'little', 'lots',
                                  'lots', 'lots']
df_train['Will_go'] = LabelEncoder().fit_transform(['+', '-', '+', '-', '-', '+', '+'])

df_train = create_df(df_train, features)
print(df_train)

df_test = {}
df_test['Looks'] = ['handsome', 'handsome', 'repulsive'] 
df_test['Alcoholic_beverage'] = ['no', 'yes', 'yes']
df_test['Eloquence'] = ['average', 'high', 'average']
df_test['Money_spent'] = ['lots', 'little', 'lots']
df_test = create_df(df_test, features)
print(df_test)

# 1.- What's the entropy So of the initial system? By system states, we mean values of binary feature "Will_go" - 0 or 1 - two states in total.

# Using Shannon's entropy formula we get 0.985, as the entropy of the initial system

p1 = df_train[df_train['Will_go'].apply(lambda will_go: will_go == 1)]['Will_go'].count()
p2 = df_train[df_train['Will_go'].apply(lambda will_go: will_go == 0)]['Will_go'].count()

h1 = df_train[df_train['Looks_handsome'].apply(lambda Looks_handsome: Looks_handsome == 1)]['Looks_handsome'].count()
h2 = df_train[df_train['Looks_handsome'].apply(lambda Looks_handsome: Looks_handsome == 0)]['Looks_handsome'].count()

train_total1 = df_train['Will_go'].count()

shannons = -((p1/train_total1)*math.log((p1/train_total1), 2)) + (-((p2/train_total1)*math.log((p2/train_total1), 2))) 

print("The entropy for the initial state is %f" % shannons)

# 2.- Let's split the data by the feature "Looks_handsome". What is the entropy  S1  of the left group - the one with "Looks_handsome". 
# What is the entropy  S2  in the opposite group? What is the information gain (IG) if we consider such a split?

# We start calculating the probability of "Will_go", for both groups (handsome or not) Then we calculate the entropy using Shannon's formula

train_handsome = df_train[df_train['Looks_handsome'].apply(lambda Looks_handsome: Looks_handsome == 1)]
train_not_handsome = df_train[df_train['Looks_handsome'].apply(lambda Looks_handsome: Looks_handsome == 0)]

handsome_will_go = train_handsome[train_handsome['Will_go'].apply(lambda will_go: will_go == 1)]['Will_go'].count()
handsome_wont_go = train_handsome[train_handsome['Will_go'].apply(lambda will_go: will_go == 0)]['Will_go'].count()

p3 = handsome_will_go/train_handsome['Will_go'].count()
p4 = handsome_wont_go/train_handsome['Will_go'].count()

not_handsome_will_go = train_not_handsome[train_not_handsome['Will_go'].apply(lambda will_go: will_go == 1)]['Will_go'].count()
not_handsome_wont_go = train_not_handsome[train_not_handsome['Will_go'].apply(lambda will_go: will_go == 0)]['Will_go'].count()

p5 = not_handsome_will_go/train_not_handsome['Will_go'].count()
p6 = not_handsome_wont_go/train_not_handsome['Will_go'].count()

e1 = -(p3*math.log(p3, 2)) + (-(p4*math.log(p4, 2)))
e2 = -(p5*math.log(p5, 2)) + (-(p6*math.log(p6, 2)))

print("The entropy is %f for handsome and %f for not handsome" % (e1, e2))

# Now, for the information gain, we we use the IG formula (IG = So - p1*S1 - p2*S2)

current_entropy = shannons - ((h1/train_total1)*e1) - ((h2/train_total1)*e2)

print("The information gain (IG) for the train set is %f" % current_entropy)

# Train a decision tree using sklearn on the training data. You may choose any depth for the tree.

y = df_train['Will_go']
df_train1, df_test1 = intersect_features(train=df_train, test=df_test)

dt = DecisionTreeClassifier(criterion='entropy', random_state=17)

dt.fit(df_train1, y);

# Additional: display the resulting tree using graphviz.

plot_tree(dt, feature_names=df_train.columns, filled=True,
         class_names=["Won't go", "Will go"]);

plt.show()

# Part 2. Functions for calculating entropy and information gain.

balls = [1 for i in range(9)] + [0 for i in range(11)]

balls_left  = [1 for i in range(8)] + [0 for i in range(5)] # 8 blue and 5 yellow
balls_right = [1 for i in range(1)] + [0 for i in range(6)] # 1 blue and 6 yellow

# Function that is used to calculate entropy

def entropy(a_list):

	a_list1 = list(a_list)
	total = len(a_list1)
	
	shannons = 0

	for i in set(a_list1):
		count = a_list1.count(i)
		p = count/total
		shannons -= (p*(math.log(p, 2)))

	return(shannons)

# 3.- What is the entropy of the state given by the list balls_left?
# The entropy for the left group of balls is aproximatly 0.96

print(entropy(balls_left))

# 4.- What is the entropy of a fair dice? (where we look at a dice as a system with 6 equally probable states)?
# For a fair dice would be aproximatly 2.58

print(entropy([1,2,3,4,5,6]))

# 5.- What is the information gain from splitting the initial dataset into balls_left and balls_right ?
# We use the IG formula used before and we got that it is 0.161

def information_gain(root, left, right):
    ''' root - initial data, left and right - two partitions of initial data'''

    so = entropy(root)
    s1 = entropy(left)
    s2 = entropy(right)

    no = len(left)/len(root)
    n1 = len(right)/len(root)

    ig = so + ((-1)*no*s1) + ((-1)*n1*s2)
    
    return ig

print("The information gain is %f" % information_gain(balls, balls_left, balls_right))

# Optional: Implement a decision tree building algorithm by calling information_gains recursively

def best_feature_to_split(X, y):
    '''Outputs information gain when splitting on best feature'''

    output = []

    for i in X.columns:
    	output.append(information_gain(y, y[X[i] == 0], y[X[i] == 1]))

    return output

print("The information gain obtained by splitting the data in all possible ways is:")
print(best_feature_to_split(df_train1, y))

# Part 3. The "Adult" dataset

data_train = pd.read_csv('./Dataset/adult_train.csv')
data_train.tail()

data_test = pd.read_csv('./Dataset/adult_test.csv')
data_test.tail()

# Necessary to remove rows with incorrect labels in test dataset
data_test = data_test[(data_test['Target'] == ' >50K.') | (data_test['Target']==' <=50K.')]

# Encode target variable as integer
data_train.loc[data_train['Target']==' <=50K', 'Target'] = 0
data_train.loc[data_train['Target']==' >50K', 'Target'] = 1

data_test.loc[data_test['Target']==' <=50K.', 'Target'] = 0
data_test.loc[data_test['Target']==' >50K.', 'Target'] = 1

print(data_test.describe(include='all').T)

fig = plt.figure(figsize=(25, 15))
cols = 5
rows = np.ceil(float(data_train.shape[1]) / cols)
for i, column in enumerate(data_train.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if data_train.dtypes[column] == np.object:
        data_train[column].value_counts().plot(kind="bar", axes=ax)
    else:
        data_train[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)
plt.show()

# We proceed to check the type of the data we are going to use

print(data_train.dtypes)
print(data_test.dtypes)

# We change the data type of Age from object to integer

data_test['Age'] = data_test['Age'].astype(int)

# We cast every float data to integer 

data_test['fnlwgt'] = data_test['fnlwgt'].astype(int)
data_test['Education_Num'] = data_test['Education_Num'].astype(int)
data_test['Capital_Gain'] = data_test['Capital_Gain'].astype(int)
data_test['Capital_Loss'] = data_test['Capital_Loss'].astype(int)
data_test['Hours_per_week'] = data_test['Hours_per_week'].astype(int)

# Fill in missing data for continuous features with their median values, for categorical features with their mode.

categorical_columns = [c for c in data_train.columns 
                       if data_train[c].dtype.name == 'object']
numerical_columns = [c for c in data_train.columns 
                     if data_train[c].dtype.name != 'object']

print('categorical_columns:', categorical_columns)
print('numerical_columns:', numerical_columns)

for c in categorical_columns:
    data_train[c].fillna(data_train[c].mode()[0], inplace=True)
    data_test[c].fillna(data_train[c].mode()[0], inplace=True)
    
for c in numerical_columns:
    data_train[c].fillna(data_train[c].median(), inplace=True)
    data_test[c].fillna(data_train[c].median(), inplace=True)

# No more missing data

print(data_train.info())

# We'll dummy code some categorical features: Workclass, Education, Martial_Status, Occupation, Relationship, Race, Sex, Country.

data_train = pd.concat([data_train[numerical_columns],
    pd.get_dummies(data_train[categorical_columns])], axis=1)

data_test = pd.concat([data_test[numerical_columns],
    pd.get_dummies(data_test[categorical_columns])], axis=1)

data_test['Country_ Holand-Netherlands'] = 0

# Now we have the holdouts

X_train = data_train.drop(['Target'], axis=1)
y_train = data_train['Target']

X_test = data_test.drop(['Target'], axis=1)
y_test = data_test['Target']

# 3.1 Decision tree without parameter tuning

# Train a decision tree (DecisionTreeClassifier) with a maximum depth of 3, and evaluate the accuracy metric on the test data. 
# Use parameter random_state = 17 for results reproducibility.

dt = DecisionTreeClassifier(max_depth = 3, random_state = 17)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

acc = accuracy_score(y_test, dt_pred)

print("The decision tree accuracy is %f. " % acc)

# 3.2 Decision tree with parameter tuning

# Train a decision tree (DecisionTreeClassifier, random_state = 17). 
# Find the optimal maximum depth using 5-fold cross-validation (GridSearchCV).

dt_parameters = {'max_depth': range(1, 20)}

dt_grid = GridSearchCV(dt, dt_parameters, cv=5, n_jobs=-1, verbose=True)
dt_grid.fit(X_train,y_train)
dt_best_params = dt_grid.best_params_

print("Best max depth using cross-validation is %d" % dt_best_params['max_depth'])

# Train a decision tree with maximum depth of 9 (it is the best max_depth in my case), and compute the test set accuracy. 
# Use parameter random_state = 17 for reproducibility.

dt = DecisionTreeClassifier(max_depth = 9, random_state = 17)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

acc = accuracy_score(y_test, dt_pred)

print("The decision tree accuracy with max_depth = 9 is %f. " % acc) # Slightly higher than the one got using max_depth = 3

# 3.3.- (Optional) Random forest without parameter tuning

# Train a random forest (RandomForestClassifier). Set the number of trees to 100 and use random_state = 17

rf = RandomForestClassifier(random_state = 17)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)

print("The random forest accuracy, using 100 trees, is %f." % rf_acc)

# 3.4.- (Optional) Random forest with parameter tuning

# Train a random forest (RandomForestClassifier). Tune the maximum depth and maximum number of features for each tree using GridSearchCV.

rf_params = {'n_estimators': range(2, 100)}
rf_grid = GridSearchCV(rf, rf_params, cv = 5, n_jobs = -1, verbose = True)
rf_grid.fit(X_train, y_train)

rf_best_params = rf_grid.best_params_

print("The number of trees that makes us get the best result is %d." % rf_best_params['n_estimators'])

# Now we calculate the score accuracy using the best value (between 1 and 99) for n_estimators.

rf = RandomForestClassifier(n_estimators = rf_best_params['n_estimators'], random_state = 17)
rf.fit(X_train, y_train)
rf_pred1 = rf.predict(X_test)

rf_acc1 = accuracy_score(y_test, rf_pred1)

print("The accuracy obtained using %d number of trees is of %f." % (rf_best_params['n_estimators'], rf_acc1))