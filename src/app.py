import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

raw_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv", sep=';')

model_data = raw_data[['contact', 'duration', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'euribor3m', 'nr.employed', 'y']]

X = model_data.drop('y', axis = 1)
y = model_data['y']

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale the data
scaler = MinMaxScaler()

# scale the data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# create a dataframe with the scaled data
X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train_df, y_train)

# extract the predictions and print out the accuracy score
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))

# We want to improve the model; first define the hyperparameters we plan to adjust
hyperparams = {
    "C": [0.001, 0.01, 1, 10, 100, 1000],
    "penalty": ["l1", "l2", "elasticnet"],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
}

#create the grid search
grid = GridSearchCV(model, hyperparams, scoring="accuracy", cv=5)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

grid.fit(X_train_df, y_train)

print(f"Best hyperparameters: {grid.best_params_}")

# create a new model using the hyperparameters suggested by the grid search 
model_grid = LogisticRegression(penalty = 'l2', C = 1000, solver = 'lbfgs')
model_grid.fit(X_train_df, y_train)

# extract the predictions and print the accuracy score.
y_grid_pred = model_grid.predict(X_test_df)
print(accuracy_score(y_test, y_grid_pred))