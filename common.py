#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:34:42 2020
@author: adityavyas
"""

#Importing the Libraries
import pandas as pd
import numpy as np
pd.options.display.max_columns = None
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from IPython.display import clear_output

#Task 1
def get_facility_data(df,num):
    """
    Fetch all the data where Federal Provider number matches NUM
    Return:
        - None if Nothing Found else a dataframe
    """
    new = df[df['Federal Provider Number'].str.match(num)]
    if len(new) == 0:
        print("No Data Found")
        return None
    return new

#Task 2 
#Subtask 1 - 
def remove_null_ns(df,name):
    """
    Input:
        df - DataFrame
        name - name of the column
    Output: 
        DataFrame
    Work : 
        Takes the name of column and the dataframe and refractor all the N's and Null
    """
    new= df[df[name] != 'N'] #Dropping Ns
    new = new.dropna(subset=[name]) #Dropping Null
    return new
#remove_null_ns(new,'Passed Quality Assurance Check')

#Subtask 2 - 
def remove_val(df,name,val):
    """
    Input:
        df - DataFrame
        name - name of the column
        val - value to be removed
    Output: 
        DataFrame
    Work : 
        Takes the name of column and the dataframe and delete the lowest
    """
    new= df[df[name] != val] #Dropping specific value
    return new

#Function to convert all the values to numeric (if found something else)
def dtype_converter(df,initial=[object],final=pd.to_numeric):
    """
    Input:
        df - DataFrame
        initial = format type e.g. [string,object,int32 etc.....]
        Final - functional reference to be applied
    Output: 
        DataFrame
    Work : 
        Finds the column with specified datatype and applies a function
    """
    for i in df.columns:
        if df[i].dtype in initial:
            df[i] = final(df[i])
    return df


def fetch_both_data():    
    """
    Input:
        None
    Output: 
        DataFrame, Dataframe
    Work : 
        Reads two dataset and returns it
    """
    nursing = pd.read_csv("COVID-19_Nursing_Home_Dataset.csv",index_col=False,low_memory=False)
    nursing['Week Ending'] = nursing['Week Ending'].astype('datetime64[ns]')
    rating = pd.read_csv("Provider_info.csv",index_col=False,low_memory=False)
    return nursing,rating


def fetch_facility(nursing,rating,num):
    """
    Input:
        nursing - DataFrame
        rating - Dataframe
        num - int
    Output: 
        DataFrame
    Work : 
        Takes the nursing dataset
        Preprocess it
        Also, fetches their ranking from Rating dataset
    """

    new = get_facility_data(nursing,num)
    new = remove_null_ns(new,'Passed Quality Assurance Check')
    new = remove_val(new,"Week Ending",min(new["Week Ending"]))
    new = new.replace('Y',1)
    new = new.replace('N',0)
    new = new.drop(['Submitted Data','Passed Quality Assurance Check','Federal Provider Number','Provider Address','Provider Name','Provider City','Provider State','Provider Zip Code','County','Geolocation'], axis = 1)
    new = new.sort_values(by='Week Ending')
    new = dtype_converter(new,initial=[object],final=pd.to_numeric)
    
    temp = get_facility_data(rating,num)
    temp = temp[['Overall Rating','Overall Rating Footnote','Health Inspection Rating','Health Inspection Rating Footnote','QM Rating','QM Rating Footnote','Staffing Rating','Staffing Rating Footnote']]

    return new, temp

"""
Feature Selection Functions
"""

def find_top_n_features(dataset,rating,rpm="Overall Rating",n=5,threshold=None):
    """
    Input:
        dataset - DataFrame (Nursing)
        rating - Dataframe (Rating)
        rpm - rating under consideration
        n = integer i.e. number of features you want
        threshold - if not provided threshold will be set to a value greater than nth feature
    Output: 
        X_transform = Transformed data
        scaler = Scaler object used 
        name_features = Top features
        threshold = Threshold value used
    """

    drp= ["Week Ending",'Submitted Data','Provider Address','Provider Name','Provider City','Provider State','Provider Zip Code','County','Geolocation']
    convert = ['Total Resident Confirmed COVID-19 Cases Per 1,000 Residents','Total Resident COVID-19 Deaths Per 1,000 Residents','Total Residents COVID-19 Deaths as a Percentage of Confirmed COVID-19 Cases','Staff Weekly Suspected COVID-19','Staff Total Suspected COVID-19','Total Number of Occupied Beds','Residents Total All Deaths']

    #Precondition Check if FPN is column or not
    assert "Federal Provider Number" in dataset.columns
    assert "Federal Provider Number" in rating.columns
    #Precondition if empty dataset
    assert len(dataset)!= 0
    assert len(rating)!= 0
    
    print("Pre-Processing dataset")
    new = dataset.replace('Y',1)
    new = new.replace('N',0)
    new = new.drop(np.intersect1d(new.columns, drp), axis = 1)
    new = new.fillna(0)
    new = new.replace([np.inf, np.nan, -np.inf], 0)
    
    #Creating y_train
    print("Creating y_train")
    y_train = [] 
    drop_index = [] #If data not found, mark that FPN
    for i in tqdm(new["Federal Provider Number"]):
        try:
            temp = get_facility_data(rating,i)[rpm].values #Fetch the star rating 
        except:
            drop_index.append(i) #If error mark the FPN for dropping
            continue
        if temp[0]==None or np.isnan(temp[0]) or temp[0]==np.inf: #If value not found mark the FPN for dropping
            drop_index.append(i)
            continue
        y_train.append(temp[0])
    y_train = pd.Series(y_train)  
    
    print("Found {} y_train values \nShape of dataset: {}\nMaintaining consistency".format(len(y_train),new.shape,))
    
    for i in drop_index:
        new.drop(new[new['Federal Provider Number'] == i].index, inplace = True) 
    print("New shape of the dataset {}".format(new.shape)) #Checking the final shape 
    
    new = new.drop(["Federal Provider Number"],axis=1)
    
    convert = ['Total Resident Confirmed COVID-19 Cases Per 1,000 Residents','Total Resident COVID-19 Deaths Per 1,000 Residents','Total Residents COVID-19 Deaths as a Percentage of Confirmed COVID-19 Cases','Staff Weekly Suspected COVID-19','Staff Total Suspected COVID-19','Total Number of Occupied Beds','Residents Total All Deaths']
    for i in convert:
        new[i] = pd.to_numeric(new[i], errors='coerce').fillna(0).astype(int)
    
    #Choosing Numeric Dataset
    print("Choosing Numeric Dataset")
    numerics = ['int16','int32','int64','float16','float32','float64']
    numerical_vars = list(new.select_dtypes(include=numerics).columns)
    data = new[numerical_vars]
    #Precondition if no numerical data found

    scaler = StandardScaler()
    scaler.fit(data.fillna(0))
    clf = LassoCV().fit(scaler.transform(data), y_train)
    importance = np.abs(clf.coef_)
    #print(importance)
    
    if not threshold:
        idx_nth = importance.argsort()[-n] #Taking nth idx
        threshold = importance[idx_nth] + 0.000000001 #Setting the threshold greater than 5th element
    
        idx_features = (-importance).argsort()[:n]
        name_features = np.array(data.columns)[idx_features]
        print("-------------------")
        print('Feature Ranking: {}'.format(name_features))
        print("-------------------")
        
    
    print("Threshold Set to {}".format(threshold))
    sfm = SelectFromModel(clf, threshold=threshold)
    sfm.fit(scaler.transform(data), y_train)
    X_transform = sfm.transform(scaler.transform(data))
    n_features = sfm.transform(scaler.transform(data)).shape[1]
    clear_output(wait=True)
    print("Final Number of Feature Selected {}".format(n_features))
    return X_transform,scaler,name_features,threshold
    


"""
#Can be used to plot a graph; consistency not verified yet; therfore it is commented out.
def plot_xtransform():
    plt.title("threshold %0.3f." % sfm.threshold)
    i = 3
    j = 4
    feature1 = X_transform[:, i]
    feature2 = X_transform[:, j]
    plt.plot(feature1, feature2, 'r.')
    plt.xlabel("First feature: {}".format(name_features[i]))
    plt.ylabel("Second feature: {}".format(name_features[j]))
    # plt.ylim([np.min(feature2), np.max(feature2)])
    plt.show()
    
"""
        
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    