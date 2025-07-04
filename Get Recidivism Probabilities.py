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
from sklearn.metrics import confusion_matrix


import CMN as CMN


rf_blackppr = []
rf_whiteppr = []
xg_whiteppr = []
xg_blackppr = []
lda_whiteppr = []
lda_blackppr = []
lr_whiteppr = []
lr_blackppr = []


rf_blackfpr = []
rf_whitefpr = []
xg_whitefpr = []
xg_blackfpr = []
lda_whitefpr = []
lda_blackfpr = []
lr_whitefpr = []
lr_blackfpr = []

rf_blackfnr = []
rf_whitefnr = []
xg_whitefnr = []
xg_blackfnr = []
lda_whitefnr = []
lda_blackfnr = []
lr_whitefnr = []
lr_blackfnr = []

rf_blacktpr = []
rf_whitetpr = []
xg_whitetpr = []
xg_blacktpr = []
lda_whitetpr = []
lda_blacktpr = []
lr_whitetpr = []
lr_blacktpr = []

rf_blacktnr = []
rf_whitetnr = []
xg_whitetnr = []
xg_blacktnr = []
lda_whitetnr = []
lda_blacktnr = []
lr_whitetnr = []
lr_blacktnr = []

rf_blacknpr = []
rf_whitenpr = []
xg_whitenpr = []
xg_blacknpr = []
lda_whitenpr = []
lda_blacknpr = []
lr_whitenpr = []
lr_blacknpr = []

rf_blackppramongpp = []
rf_whiteppramongpp = []
xg_whiteppramongpp = []
xg_blackppramongpp = []
lda_whiteppramongpp = []
lda_blackppramongpp = []
lr_whiteppramongpp = []
lr_blackppramongpp = []


rf_blacknpramongnp = []
rf_whitenpramongnp = []
xg_whitenpramongnp = []
xg_blacknpramongnp = []
lda_whitenpramongnp = []
lda_blacknpramongnp = []
lr_whitenpramongnp = []
lr_blacknpramongnp = []

rf_blackacc= []
rf_whiteacc = []
xg_whiteacc= []
xg_blackacc = []
lda_whiteacc= []
lda_blackacc = []
lr_whiteacc = []
lr_blackacc = []






lr_accuracy = []
lda_accuracy = []
rf_accuracy = []
dtree_accuracy = []
xg_accuracy = []
test_array = []
for test_fold in range(0, 250):

    print('start loop')

    test_fold_title = 'Folds/Fold ' + str(test_fold) + '.csv'
    test_fold_prob = 'Folds/Probability Fold ' + str(test_fold) + '.csv'

    test = pd.read_csv(test_fold_title)
    y_test = test['Recidivism_Within_3years']
    y_train = test.drop(['Unnamed: 0', 'Recidivism_Within_3years'], axis = 1)
    columns = y_train.columns.tolist()

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
   

    rf = RandomForestClassifier(n_estimators = 100, random_state=22)
    rf.fit(X_train, X_test)
    print('check 2')

        #Build the model
    xg = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=1, objective='binary:logistic', random_state=22)

    # Trained the model
    xg.fit(X_train, X_test)

    test['Logistic Regression Probability Predictions'] = CMN.extract_positive_class_probabilities(lr.predict_proba(y_train[columns]))
    test['LDA Probability Predictions'] = CMN.extract_positive_class_probabilities(lda.predict_proba(y_train[columns]))
    test['RF Probability Predictions'] = CMN.extract_positive_class_probabilities(rf.predict_proba(y_train[columns]))
    test['XG Probability Predictions'] = CMN.extract_positive_class_probabilities(xg.predict_proba(y_train[columns]))

    test_array.append(test)


df = pd.DataFrame()
df = pd.concat(test_array)






df.to_csv('Predicted Probabilities of Recidivism (Stats for Probability Testing).csv')