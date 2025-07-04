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

for test_fold in range(0, 250):

    print('start loop')

    test_fold_title = 'Folds/Fold ' + str(test_fold) + '.csv'

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

    rf_blackppr.append(CMN.group_one_ppr(y_train, 'Race_BLACK', y_test, rf.predict(y_train[columns])))
    xg_blackppr.append(CMN.group_one_ppr(y_train, 'Race_BLACK', y_test, xg.predict(y_train[columns])))
    lda_blackppr.append(CMN.group_one_ppr(y_train, 'Race_BLACK', y_test, lda.predict(y_train[columns])))
    lr_blackppr.append(CMN.group_one_ppr(y_train, 'Race_BLACK', y_test, lr.predict(y_train[columns])))

    rf_whiteppr.append(CMN.group_one_ppr(y_train, 'Race_WHITE', y_test, rf.predict(y_train[columns])))
    xg_whiteppr.append(CMN.group_one_ppr(y_train, 'Race_WHITE', y_test, xg.predict(y_train[columns])))
    lr_whiteppr.append(CMN.group_one_ppr(y_train, 'Race_WHITE', y_test, lr.predict(y_train[columns])))
    lda_whiteppr.append(CMN.group_one_ppr(y_train, 'Race_WHITE', y_test, lda.predict(y_train[columns])))

    rf_blacknpr.append(CMN.group_one_npr(y_train, 'Race_BLACK', y_test, rf.predict(y_train[columns])))
    xg_blacknpr.append(CMN.group_one_npr(y_train, 'Race_BLACK', y_test, xg.predict(y_train[columns])))
    lda_blacknpr.append(CMN.group_one_npr(y_train, 'Race_BLACK', y_test, lda.predict(y_train[columns])))
    lr_blacknpr.append(CMN.group_one_npr(y_train, 'Race_BLACK', y_test, lr.predict(y_train[columns])))

    rf_whitenpr.append(CMN.group_one_npr(y_train, 'Race_WHITE', y_test, rf.predict(y_train[columns])))
    xg_whitenpr.append(CMN.group_one_npr(y_train, 'Race_WHITE', y_test, xg.predict(y_train[columns])))
    lr_whitenpr.append(CMN.group_one_npr(y_train, 'Race_WHITE', y_test, lr.predict(y_train[columns])))
    lda_whitenpr.append(CMN.group_one_npr(y_train, 'Race_WHITE', y_test, lda.predict(y_train[columns])))

    rf_blackfnr.append(CMN.group_one_fn(y_train, 'Race_BLACK', y_test, rf.predict(y_train[columns])))
    xg_blackfnr.append(CMN.group_one_fn(y_train, 'Race_BLACK', y_test, xg.predict(y_train[columns])))
    lda_blackfnr.append(CMN.group_one_fn(y_train, 'Race_BLACK', y_test, lda.predict(y_train[columns])))
    lr_blackfnr.append(CMN.group_one_fn(y_train, 'Race_BLACK', y_test, lr.predict(y_train[columns])))

    rf_whitefnr.append(CMN.group_one_fn(y_train, 'Race_WHITE', y_test, rf.predict(y_train[columns])))
    xg_whitefnr.append(CMN.group_one_fn(y_train, 'Race_WHITE', y_test, xg.predict(y_train[columns])))
    lr_whitefnr.append(CMN.group_one_fn(y_train, 'Race_WHITE', y_test, lr.predict(y_train[columns])))
    lda_whitefnr.append(CMN.group_one_fn(y_train, 'Race_WHITE', y_test, lda.predict(y_train[columns])))

    rf_blacktnr.append(CMN.group_one_tn(y_train, 'Race_BLACK', y_test, rf.predict(y_train[columns])))
    xg_blacktnr.append(CMN.group_one_tn(y_train, 'Race_BLACK', y_test, xg.predict(y_train[columns])))
    lda_blacktnr.append(CMN.group_one_tn(y_train, 'Race_BLACK', y_test, lda.predict(y_train[columns])))
    lr_blacktnr.append(CMN.group_one_tn(y_train, 'Race_BLACK', y_test, lr.predict(y_train[columns])))

    rf_whitetnr.append(CMN.group_one_tn(y_train, 'Race_WHITE', y_test, rf.predict(y_train[columns])))
    xg_whitetnr.append(CMN.group_one_tn(y_train, 'Race_WHITE', y_test, xg.predict(y_train[columns])))
    lr_whitetnr.append(CMN.group_one_tn(y_train, 'Race_WHITE', y_test, lr.predict(y_train[columns])))
    lda_whitetnr.append(CMN.group_one_tn(y_train, 'Race_WHITE', y_test, lda.predict(y_train[columns])))


    rf_blacktpr.append(CMN.group_one_tp(y_train, 'Race_BLACK', y_test, rf.predict(y_train[columns])))
    xg_blacktpr.append(CMN.group_one_tp(y_train, 'Race_BLACK', y_test, xg.predict(y_train[columns])))
    lda_blacktpr.append(CMN.group_one_tp(y_train, 'Race_BLACK', y_test, lda.predict(y_train[columns])))
    lr_blacktpr.append(CMN.group_one_tp(y_train, 'Race_BLACK', y_test, lr.predict(y_train[columns])))

    rf_whitetpr.append(CMN.group_one_tp(y_train, 'Race_WHITE', y_test, rf.predict(y_train[columns])))
    xg_whitetpr.append(CMN.group_one_tp(y_train, 'Race_WHITE', y_test, xg.predict(y_train[columns])))
    lr_whitetpr.append(CMN.group_one_tp(y_train, 'Race_WHITE', y_test, lr.predict(y_train[columns])))
    lda_whitetpr.append(CMN.group_one_tp(y_train, 'Race_WHITE', y_test, lda.predict(y_train[columns])))

    
    rf_blackfpr.append(CMN.group_one_fp(y_train, 'Race_BLACK', y_test, rf.predict(y_train[columns])))
    xg_blackfpr.append(CMN.group_one_fp(y_train, 'Race_BLACK', y_test, xg.predict(y_train[columns])))
    lda_blackfpr.append(CMN.group_one_fp(y_train, 'Race_BLACK', y_test, lda.predict(y_train[columns])))
    lr_blackfpr.append(CMN.group_one_fp(y_train, 'Race_BLACK', y_test, lr.predict(y_train[columns])))

    rf_whitefpr.append(CMN.group_one_fp(y_train, 'Race_WHITE', y_test, rf.predict(y_train[columns])))
    xg_whitefpr.append(CMN.group_one_fp(y_train, 'Race_WHITE', y_test, xg.predict(y_train[columns])))
    lr_whitefpr.append(CMN.group_one_fp(y_train, 'Race_WHITE', y_test, lr.predict(y_train[columns])))
    lda_whitefpr.append(CMN.group_one_fp(y_train, 'Race_WHITE', y_test, lda.predict(y_train[columns])))

    rf_blackppramongpp.append(CMN.ppr_among_pp(y_train, 'Race_BLACK', y_test, rf.predict(y_train[columns])))
    xg_blackppramongpp.append(CMN.ppr_among_pp(y_train, 'Race_BLACK', y_test, xg.predict(y_train[columns])))
    lda_blackppramongpp.append(CMN.ppr_among_pp(y_train, 'Race_BLACK', y_test, lda.predict(y_train[columns])))
    lr_blackppramongpp.append(CMN.ppr_among_pp(y_train, 'Race_BLACK', y_test, lr.predict(y_train[columns])))

    rf_whiteppramongpp.append(CMN.ppr_among_pp(y_train, 'Race_WHITE', y_test, rf.predict(y_train[columns])))
    xg_whiteppramongpp.append(CMN.ppr_among_pp(y_train, 'Race_WHITE', y_test, xg.predict(y_train[columns])))
    lr_whiteppramongpp.append(CMN.ppr_among_pp(y_train, 'Race_WHITE', y_test, lr.predict(y_train[columns])))
    lda_whiteppramongpp.append(CMN.ppr_among_pp(y_train, 'Race_WHITE', y_test, lda.predict(y_train[columns])))

    rf_blacknpramongnp.append(CMN.npr_among_np(y_train, 'Race_BLACK', y_test, rf.predict(y_train[columns])))
    xg_blacknpramongnp.append(CMN.npr_among_np(y_train, 'Race_BLACK', y_test, xg.predict(y_train[columns])))
    lda_blacknpramongnp.append(CMN.npr_among_np(y_train, 'Race_BLACK', y_test, lda.predict(y_train[columns])))
    lr_blacknpramongnp.append(CMN.npr_among_np(y_train, 'Race_BLACK', y_test, lr.predict(y_train[columns])))

    rf_whitenpramongnp.append(CMN.npr_among_np(y_train, 'Race_WHITE', y_test, rf.predict(y_train[columns])))
    xg_whitenpramongnp.append(CMN.npr_among_np(y_train, 'Race_WHITE', y_test, xg.predict(y_train[columns])))
    lr_whitenpramongnp.append(CMN.npr_among_np(y_train, 'Race_WHITE', y_test, lr.predict(y_train[columns])))
    lda_whitenpramongnp.append(CMN.npr_among_np(y_train, 'Race_WHITE', y_test, lda.predict(y_train[columns])))

    rf_blackacc.append(CMN.group_acc(y_train, 'Race_BLACK', y_test, rf.predict(y_train[columns])))
    xg_blackacc.append(CMN.group_acc(y_train, 'Race_BLACK', y_test, xg.predict(y_train[columns])))
    lda_blackacc.append(CMN.group_acc(y_train, 'Race_BLACK', y_test, lda.predict(y_train[columns])))
    lr_blackacc.append(CMN.group_acc(y_train, 'Race_BLACK', y_test, lr.predict(y_train[columns])))

    rf_whiteacc.append(CMN.group_acc(y_train, 'Race_WHITE', y_test, rf.predict(y_train[columns])))
    xg_whiteacc.append(CMN.group_acc(y_train, 'Race_WHITE', y_test, xg.predict(y_train[columns])))
    lr_whiteacc.append(CMN.group_acc(y_train, 'Race_WHITE', y_test, lr.predict(y_train[columns])))
    lda_whiteacc.append(CMN.group_acc(y_train, 'Race_WHITE', y_test, lda.predict(y_train[columns])))





    



    rf_accuracy.append(accuracy_score(rf.predict(y_train[columns]), y_test))
    lda_accuracy.append(accuracy_score(lda.predict(y_train[columns]), y_test))
    xg_accuracy.append(accuracy_score(xg.predict(y_train[columns]), y_test))
    lr_accuracy.append(accuracy_score(lr.predict(y_train[columns]), y_test))
    print('check 3')


df = pd.DataFrame()

df['Accuracy of Logistic Regression Algorithm'] = lr_accuracy

df['Accuracy of LDA Algorithm'] = lda_accuracy

df['Accuracy of XgBoost'] = xg_accuracy


df['Accuracy of Random Forest'] = rf_accuracy

df['Black PPR RF'] = rf_blackppr
df['Black PPR XG'] = xg_blackppr
df['Black PPR LDA'] = lda_blackppr
df['Black PPR LR'] = lr_blackppr

df['White PPR RF'] = rf_whiteppr
df['White PPR XG'] = xg_whiteppr
df['White PPR LDA'] = lda_whiteppr
df['White PPR LR'] = lr_whiteppr

df['Black NPR RF'] = rf_blacknpr
df['Black NPR XG'] = xg_blacknpr
df['Black NPR LDA'] = lda_blacknpr
df['Black NPR LR'] = lr_blacknpr

df['White NPR RF'] = rf_whitenpr
df['White NPR XG'] = xg_whitenpr
df['White NPR LDA'] = lda_whitenpr
df['White NPR LR'] = lr_whitenpr

df['Black TPR RF'] = rf_blacktpr
df['Black TPR XG'] = xg_blacktpr
df['Black TPR LDA'] = lda_blacktpr
df['Black TPR LR'] = lr_blacktpr

df['White TPR RF'] = rf_whitetpr
df['White TPR XG'] = xg_whitetpr
df['White TPR LDA'] = lda_whitetpr
df['White TPR LR'] = lr_whitetpr

df['Black FPR RF'] = rf_blackfpr
df['Black FPR XG'] = xg_blackfpr
df['Black FPR LDA'] = lda_blackfpr
df['Black FPR LR'] = lr_blackfpr

df['White FPR RF'] = rf_whitefpr
df['White FPR XG'] = xg_whitefpr
df['White FPR LDA'] = lda_whitefpr
df['White FPR LR'] = lr_whitefpr


df['Black TNR RF'] = rf_blacktnr
df['Black TNR XG'] = xg_blacktnr
df['Black TNR LDA'] = lda_blacktnr
df['Black TNR LR'] = lr_blacktnr

df['White TNR RF'] = rf_whitetnr
df['White TNR XG'] = xg_whitetnr
df['White TNR LDA'] = lda_whitetnr
df['White TNR LR'] = lr_whitetnr

df['Black FNR RF'] = rf_blackfnr
df['Black FNR XG'] = xg_blackfnr
df['Black FNR LDA'] = lda_blackfnr
df['Black FNR LR'] = lr_blackfnr

df['White FNR RF'] = rf_whitefnr
df['White FNR XG'] = xg_whitefnr
df['White FNR LDA'] = lda_whitefnr
df['White FNR LR'] = lr_whitefnr





df['Black PPR among PP RF'] = rf_blackppramongpp
df['Black PPR among PP XG'] = xg_blackppramongpp
df['Black PPR among PP LDA'] = lda_blackppramongpp
df['Black PPR among PP LR'] = lr_blackppramongpp

df['White PPR among PP RF'] = rf_whiteppramongpp
df['White PPR among PP XG'] = xg_whiteppramongpp
df['White PPR among PP LDA'] = lda_whiteppramongpp
df['White PPR among PP LR'] = lr_whiteppramongpp

df['Black NPR among NP RF'] = rf_blacknpramongnp
df['Black NPR among NP XG'] = xg_blacknpramongnp
df['Black NPR among NP LDA'] = lda_blacknpramongnp
df['Black NPR among NP LR'] = lr_blacknpramongnp

df['White NPR among NP RF'] = rf_whitenpramongnp
df['White NPR among NP XG'] = xg_whitenpramongnp
df['White NPR among NP LDA'] = lda_whitenpramongnp
df['White NPR among NP LR'] = lr_whitenpramongnp



df['Black Accuracy under Random Forest'] = rf_blackacc
df['White Accuracy under Random Forest'] = rf_whiteacc 
df['White Accuracy under XG'] = xg_whiteacc
df['Black Accuracy under XG'] = xg_blackacc 

df['White Accuracy under LDA'] = lda_whiteacc
df['Black Accuracy under LDA'] = lda_blackacc 
df['White Accuracy under LR'] = lr_whiteacc
df['Black Accuracy under LR'] = lr_blackacc 









df.to_csv('Fairness Metrics - All Predicted Outcome Binary Metrics from Verma Paper.csv')