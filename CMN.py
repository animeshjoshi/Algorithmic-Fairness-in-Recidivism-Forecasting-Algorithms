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
from sklearn.metrics import accuracy_score

def group_one_ppr(df, protected_class, ground_truth, predictions):
    
    df['Ground Truth'] = ground_truth
    df['Predictions'] = predictions
    groupone_df = df[df[protected_class] == 1]
    
    #cm = confusion_matrix(groupone_df['Ground Truth'], groupone_df['Predictions'])
    

    return len(groupone_df[groupone_df['Predictions'] == 0])/len(groupone_df)
   
    


def group_two_ppr(df, protected_class, ground_truth, predictions):
    
    df['Ground Truth'] = ground_truth
    df['Predictions'] = predictions
    groupone_df = df[df[protected_class] == 0]
   
    
    return len(groupone_df[groupone_df['Predictions'] == 0])/len(groupone_df)

def group_one_npr(df, protected_class, ground_truth, predictions):
    
    df['Ground Truth'] = ground_truth
    df['Predictions'] = predictions
    groupone_df = df[df[protected_class] == 1]
   
    return len(groupone_df[groupone_df['Predictions'] == 1])/len(groupone_df)

def group_two_npr(df, protected_class, ground_truth, predictions):
    
    df['Ground Truth'] = ground_truth
    df['Predictions'] = predictions
    groupone_df = df[df[protected_class] == 0]
    print(groupone_df['Predictions'])
    
    return len(groupone_df[groupone_df['Predictions'] == 1])/len(groupone_df)



def group_one_fn(df, protected_class, ground_truth, predictions):
    
    df['Ground Truth'] = ground_truth
    df['Predictions'] = predictions
    groupone_df = df[df[protected_class] == 1]
    
    cm = confusion_matrix(groupone_df['Ground Truth'], groupone_df['Predictions'])
    tp = cm[0,0]
    fp = cm[0,1]
    tn = cm[1,0]
    fn = cm[1,1]

    sum = tp + fp + tn + fn
    
    df = df.drop(['Ground Truth', 'Predictions'], axis =1 )
    return len(groupone_df[(groupone_df['Predictions'] == 1) & (groupone_df['Ground Truth'] == 0)])/len(groupone_df)





def group_one_fp(df, protected_class, ground_truth, predictions):
    
    df['Ground Truth'] = ground_truth
    df['Predictions'] = predictions
    groupone_df = df[df[protected_class] == 1]
    cm = confusion_matrix(groupone_df['Ground Truth'], groupone_df['Predictions'])
    tp = cm[0,0]
    fp = cm[0,1]
    tn = cm[1,0]
    fn = cm[1,1]

    sum = tp + fp + tn + fn
    
    df = df.drop(['Ground Truth', 'Predictions'], axis =1 )
    return len(groupone_df[(groupone_df['Predictions'] == 0) & (groupone_df['Ground Truth'] == 1)])/len(groupone_df)


def group_one_tn(df, protected_class, ground_truth, predictions):
    
    df['Ground Truth'] = ground_truth
    df['Predictions'] = predictions
    groupone_df = df[df[protected_class] == 1]
    cm = confusion_matrix(groupone_df['Ground Truth'], groupone_df['Predictions'])
    tp = cm[0,0]
    fp = cm[0,1]
    tn = cm[1,0]
    fn = cm[1,1]

    sum = tp + fp + tn + fn
    df = df.drop(['Ground Truth', 'Predictions'], axis =1 )
    
    return len(groupone_df[(groupone_df['Predictions'] == 1) & (groupone_df['Ground Truth'] == 1)])/len(groupone_df)
    




def group_one_tp(df, protected_class, ground_truth, predictions):
    
    df['Ground Truth'] = ground_truth
    df['Predictions'] = predictions
    groupone_df = df[df[protected_class] == 1]
    cm = confusion_matrix(groupone_df['Ground Truth'], groupone_df['Predictions'])
    tp = cm[0,0]
    fp = cm[0,1]
    tn = cm[1,0]
    fn = cm[1,1]

    sum = tp + fp + tn + fn
    df = df.drop(['Ground Truth', 'Predictions'], axis =1 )
    
    return len(groupone_df[(groupone_df['Predictions'] == 0) & (groupone_df['Ground Truth'] == 0)])/len(groupone_df)



def ppr_among_pp(df, protected_class, ground_truth, predictions):

    df['Ground Truth'] = ground_truth
    df['Predictions'] = predictions
    df = df[df['Predictions'] == 0]
    groupone_df = df[df[protected_class] == 1]
    
    

    result = len(groupone_df[groupone_df['Ground Truth'] == 0])/len(groupone_df)

    df = df.drop(['Ground Truth', 'Predictions'], axis =1 )
    
    return len(groupone_df[(groupone_df['Predictions'] == 0)])/len(groupone_df)

def npr_among_np(df, protected_class, ground_truth, predictions):

    df['Ground Truth'] = ground_truth
    df['Predictions'] = predictions
    df = df[df['Predictions'] == 1]
    groupone_df = df[df[protected_class] == 1]
    
    cm = confusion_matrix(groupone_df['Ground Truth'], groupone_df['Predictions'])
   
    result = len(groupone_df[groupone_df['Ground Truth'] == 1])/len(groupone_df)

    df = df.drop(['Ground Truth', 'Predictions'], axis =1 )
    
    return len(groupone_df[(groupone_df['Predictions'] == 1)])/len(groupone_df)

def group_acc(df, protected_class, ground_truth, predictions):

    df['Ground Truth'] = ground_truth
    df['Predictions'] = predictions

    groupone_df = df[df[protected_class] == 1]
    


 
   
    result = accuracy_score(groupone_df['Predictions'], groupone_df['Ground Truth'])

    df = df.drop(['Ground Truth', 'Predictions'], axis =1 )
    
    return result


def extract_positive_class_probabilities(predict_proba):
    arr = []
    for x in predict_proba:

        arr.append(x[0])

    return arr