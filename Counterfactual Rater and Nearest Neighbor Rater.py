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
from scipy.spatial import distance

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

    flipped_columns = test['Race_WHITE']
    y_train = test.drop(['Unnamed: 0', 'Recidivism_Within_3years', 'Race_WHITE'], axis = 1)
  
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

    
    #flipped_columns = train_data['Race_WHITE']
    X_train = train_data.drop(['Recidivism_Within_3years', 'Race_WHITE'], axis = 1)
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

    
    print('hi')

    data = y_train[columns]
    distance_data = data.drop('Race_BLACK', axis = 1)
    lr_original_predictions = []
    lr_nearest_neighbor = []
    lda_original_predictions = []
    lda_nearest_neighbor = []
    rf_original_predictions = []
    rf_nearest_neighbor = []
    xg_original_predictions = []
    xg_nearest_neighbor = []
    index = 0
    for x in range(0, len(distance_data)):
        print(x)
        if data['Race_BLACK'][x] == 1:
            current_observation = distance_data.iloc[x].to_numpy()

            min_distance = 10000000

            for y in range(0, len(distance_data)):
                if data['Race_BLACK'][y] == 0: 
                    if x != y:
                        cov = np.cov(current_observation, distance_data.iloc[y].to_numpy())
                    

                        e = current_observation - distance_data.iloc[y].to_numpy()
                        X = np.vstack([current_observation,distance_data.iloc[y].to_numpy()])
                        V = np.cov(X.T) 
                        p = np.linalg.pinv(V)
                        dist = np.sqrt(np.dot(np.dot(e, p), e.T))
                       
                        if dist < min_distance:

                            min_distance = dist
                            index = y

            lr_original_predictions.append(lr.predict(data.iloc[x].to_numpy().reshape(1, -1)))
            lr_nearest_neighbor.append(lr.predict(data.iloc[index].to_numpy().reshape(1, -1)))
            lda_original_predictions.append(lda.predict(data.iloc[x].to_numpy().reshape(1, -1)))
            lda_nearest_neighbor.append(lda.predict(data.iloc[index].to_numpy().reshape(1, -1)))
            xg_original_predictions.append(xg.predict(data.iloc[x].to_numpy().reshape(1, -1)))
            xg_nearest_neighbor.append(xg.predict(data.iloc[index].to_numpy().reshape(1, -1)))
            rf_original_predictions.append(rf.predict(data.iloc[x].to_numpy().reshape(1, -1)))
            rf_nearest_neighbor.append(rf.predict(data.iloc[index].to_numpy().reshape(1, -1)))

        if data['Race_BLACK'][x] == 0:
            current_observation = distance_data.iloc[x].to_numpy()

            min_distance = 10000000

            for y in range(0, len(distance_data)):
                if data['Race_BLACK'][y] == 1: 
                    if x != y:
                   
                       
                        e = current_observation - distance_data.iloc[y].to_numpy()
                        X = np.vstack([current_observation,distance_data.iloc[y].to_numpy()])
                        V = np.cov(X.T) 
                        p = np.linalg.pinv(V)
                        dist = np.sqrt(np.dot(np.dot(e, p), e.T))

                        if dist < min_distance:

                            min_distance = dist
                            index = y

            lr_original_predictions.append(lr.predict(data.iloc[x].to_numpy().reshape(1, -1)))
            lr_nearest_neighbor.append(lr.predict(data.iloc[index].to_numpy().reshape(1, -1)))
            lda_original_predictions.append(lda.predict(data.iloc[x].to_numpy().reshape(1, -1)))
            lda_nearest_neighbor.append(lda.predict(data.iloc[index].to_numpy().reshape(1, -1)))
            xg_original_predictions.append(xg.predict(data.iloc[x].to_numpy().reshape(1, -1)))
            xg_nearest_neighbor.append(xg.predict(data.iloc[index].to_numpy().reshape(1, -1)))
            rf_original_predictions.append(rf.predict(data.iloc[x].to_numpy().reshape(1, -1)))
            rf_nearest_neighbor.append(rf.predict(data.iloc[index].to_numpy().reshape(1, -1)))
            

            

    y_train['Race_BLACK'] = flipped_columns

    test['Logistic Regression Counterfactual Predictions'] = lr.predict(y_train[columns])
    test['LDA Counterfactual Predictions'] = lda.predict(y_train[columns])
    test['RF Counterfactual Predictions'] = rf.predict(y_train[columns])
    test['XG Counterfactual Predictions'] = xg.predict(y_train[columns])

    test['Logistic Original Predictions'] = lr_original_predictions
    test['RF Original Predictions'] = rf_original_predictions
    test['XG Original Predictions'] = xg_original_predictions
    test['LDA Original Predictions'] = lda_original_predictions

    test['Logistic NN Predictions'] = lr_nearest_neighbor
    test['RF NN Predictions'] = rf_nearest_neighbor
    test['XG NN Predictions'] = xg_nearest_neighbor
    test['LDA NN Predictions'] = lda_nearest_neighbor



    test_array.append(test)


df = pd.DataFrame()
df = pd.concat(test_array)






df.to_csv('Counterfactuals and Nearest Neighbors (Stats for Similarity Testing).csv')