import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

lr_accuracy = []
lda_accuracy = []
rf_accuracy = []
dtree_accuracy = []
xg_accuracy = []

for test_fold in range(0, 250):

    print('start loop')

    test_fold_title = 'Folds/Fold ' + str(test_fold) + '.csv'

    test = pd.read_csv(test_fold_title)
    y_test = test['Recidivism_Within_3years']
    y_train = test.drop(['Unnamed: 0', 'Recidivism_Within_3years'], axis = 1)

    train_arr = []

    
    for train_titles in range(0, 250):

        train_fold_title = 'Folds/Fold ' + str(train_titles) + '.csv'

        print(train_fold_title)
        print(test_fold_title)

        print(train_fold_title != test_fold_title)

        if train_fold_title != test_fold_title:

            print('hi')

            train = pd.read_csv(train_fold_title)
            train = train.drop('Unnamed: 0', axis = 1)

            train_arr.append(train)

    train_data = pd.concat(train_arr)

    X_train = train_data.drop('Recidivism_Within_3years', axis = 1)
    X_test = train_data['Recidivism_Within_3years']

    lr = LogisticRegression()
    lr.fit(X_train, X_test)
    lda = LinearDiscriminantAnalysis()
    print('check')
    lda.fit(X_train, X_test)
    print('check 1.005')
   
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, X_test)
    rf = RandomForestClassifier(n_estimators = 100, random_state=22)
    rf.fit(X_train, X_test)
    print('check 2')

        #Build the model
    xg = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=1, objective='binary:logistic', random_state=22)

    # Trained the model
    xg.fit(X_train, X_test)

    rf_accuracy.append(accuracy_score(rf.predict(y_train), y_test))
    lda_accuracy.append(accuracy_score(lda.predict(y_train), y_test))
    xg_accuracy.append(accuracy_score(xg.predict(y_train), y_test))
    dtree_accuracy.append(accuracy_score(dtree.predict(y_train), y_test))
    lr_accuracy.append(accuracy_score(lr.predict(y_train), y_test))
    print('check 3')


df = pd.DataFrame()

df['Logistic Regression Accuracy'] = lr_accuracy

df['LDA Accuracy'] = lda_accuracy

df['KNN Accuracy'] = xg_accuracy


df['Decision Tree Accuracy'] = dtree_accuracy

df['Random Forest Accuracy'] = rf_accuracy


df.to_csv('Model Comparison.csv')