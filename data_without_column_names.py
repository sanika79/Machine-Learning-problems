# load the dataset

# example of using the ColumnTransformer for the Abalone dataset
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/abalone.csv'


dataframe = read_csv(url, header=None)
print(dataframe.head(5))

## The first 5 rows of the data

#    0      1      2      3       4       5       6      7   8
# 0  M  0.455  0.365  0.095  0.5140  0.2245  0.1010  0.150  15
# 1  M  0.350  0.265  0.090  0.2255  0.0995  0.0485  0.070   7
# 2  F  0.530  0.420  0.135  0.6770  0.2565  0.1415  0.210   9
# 3  M  0.440  0.365  0.125  0.5160  0.2155  0.1140  0.155  10
# 4  I  0.330  0.255  0.080  0.2050  0.0895  0.0395  0.055   7
# ...



# split into inputs and outputs
last_ix = len(dataframe.columns) - 1
X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
print(X.shape, y.shape)


# determine categorical and numerical features
numerical_ix = X.select_dtypes(include=['int64', 'float64']).columns
categorical_ix = X.select_dtypes(include=['object', 'bool']).columns

print(numerical_ix)
print(categorical_ix)


# define the data preparation for the columns
t = [('cat', OneHotEncoder(), categorical_ix), ('num', MinMaxScaler(), numerical_ix)]
col_transform = ColumnTransformer(transformers=t)

# define the model
model = SVR(kernel='rbf',gamma='scale',C=100)

# define the data preparation and modeling pipeline
pipeline = Pipeline(steps=[('prep',col_transform), ('m', model)])

# define the model cross-validation configuration
cv = KFold(n_splits=10, shuffle=True, random_state=1)

# evaluate the pipeline using cross validation and calculate MAE
scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

# convert MAE scores to positive values
scores = absolute(scores)
# summarize the model performance
print('MAE: %.3f (%.3f)' % (mean(scores), std(scores)))





