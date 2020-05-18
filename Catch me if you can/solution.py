import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score
import numpy as np
import sklearn.preprocessing as preprocessing

# Function for writing predictions to a file. Taken from https://www.kaggle.com/kashnitsky/alice-logistic-regression-baseline

def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


def iniciate_time(df, X_taken):
	hour = df['time1'].apply(lambda ts: ts.hour)
	morning = ((hour >= 6) & (hour <= 11)).astype('int')
	day = ((hour >= 12) & (hour <= 18)).astype('int')
	evening = ((hour >= 19) & (hour <= 21)).astype('int')
	night = ((hour >= 22) & (hour < 24)).astype('int')
	midnight = ((hour >= 0) & (hour <= 5)).astype('int')
	X = hstack([X_taken, morning.values.reshape(-1, 1), 
						 day.values.reshape(-1, 1),
						 evening.values.reshape(-1, 1),
						 night.values.reshape(-1, 1),
						 midnight.values.reshape(-1, 1)])
	return X

# Solution for the competition "Catch Me If You Can ("Alice")", from Kaggle. 
# Link to the competition: https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/overview

# First we charge the data

df_train = pd.read_csv("./train_sessions.csv")
df_test = pd.read_csv("./test_sessions.csv")

X_train = df_train
X_test = df_test

# List that contain the two groups of features

times = []
sites = []
for i in range (1, 11):
	sites.append('site' + str(i))
	times.append('time' + str(i))

# change the times from string to datetime

X_train[times] = X_train[times].fillna(0).apply(pd.to_datetime)
X_test[times] = X_test[times].fillna(0).apply(pd.to_datetime)

# Save the information of the sites as a text file.

X_train_text = X_train[sites].fillna(0).astype('int').to_csv('train_set_text.txt', sep= ' ', index=None, header=None)
X_test_text = X_test[sites].fillna(0).astype('int').to_csv('test_set_text.txt', sep= ' ', index=None, header=None)

# Use count vectorizer to apply BoW (Bag of Words)

cv = CountVectorizer()
with open('train_set_text.txt') as train_set:
	X_train = cv.fit_transform(train_set)
with open('test_set_text.txt') as test_set:
	X_test = cv.transform(test_set)

# Create the Y set for the training set.

Y_train = df_train['target']

# Create and cross validate the Logistic Regression

lr = LogisticRegression(random_state=17, max_iter=500, n_jobs=-1)
cv_lr = cross_val_score(lr, X_train, Y_train, cv=5, scoring='roc_auc')

print(cv_lr)

# Add information that says the part of the day where the first session got place.

X_train_using_time = iniciate_time(df_train.fillna(0), X_train)
X_test_using_time = iniciate_time(df_test.fillna(0), X_test)

# Train the Logistic Regression

lr.fit(X_train_using_time, Y_train)
lr_predict = lr.predict(X_test_using_time)

print(lr_predict)

# Now we store the prediction for submitting.

write_to_submission_file(lr_predict, 'submition.txt')