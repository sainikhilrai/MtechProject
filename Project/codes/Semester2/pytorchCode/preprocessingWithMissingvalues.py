
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import calendar
import copy
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler
import calendar
import datetime
from datetime import datetime


def preprocess(path):
    
    #read the data
    car_df= pd.read_csv(path)

    #gives the information about the dataset and its features.
    car_df.info()

    featureName= list(car_df.columns.values) #get the column/feature name of the data
    #prints the frequency of types in features
    '''
    for i in range(len(featureName)):
        print("\nFeature Name: ",featureName[i])
        print(car_df[featureName[i]].value_counts())
    '''
    #handle the ambiguity in the data
    dayIndex= car_df.shape[0] + 2   #set the value other than 0-15419
    monthIndex= car_df.shape[0] + 1 #set the value other than 0-15419

    for i in range(car_df.shape[0]):
        if(car_df['DayOfWeekClaimed'][i]=='0'):
            dayIndex= i
            print("index: ", i)
        if(car_df['MonthClaimed'][i]=='0'):
            monthIndex= i
            print("\nindex:",i)

    #create dataframe for keeping new form of dataset
    carData= pd.DataFrame(columns=featureName)
    j= 0
    #put all the data into new dataframe
    for i in range(car_df.shape[0]):
        if(i!=dayIndex):
            carData.loc[j]= car_df.loc[i]
            j += 1

    #replace map
    replace_Month = {'Month':{'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
                              'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}}

    replace_MonthClaimed = {'MonthClaimed':{'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
                              'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}}

    carData.replace(replace_Month,inplace=True)
    carData.replace(replace_MonthClaimed,inplace=True)

    #function to calculate the no of days passed between the accident and the claims.
    # Reporting Gap:
    def get_day(year,month,weekOfMonth,dayOfWeek):
        count = 0
        c = calendar.TextCalendar(calendar.SUNDAY)
        l = []
        year= int(year)
        month=int(month)
        for i in c.itermonthdates(year,month):
            l.append(i)
        for j in range(len(l)):
            day = calendar.day_name[l[j].weekday()]
            if day == dayOfWeek:
                count += 1
            if count == weekOfMonth:
                return l[j]
                break

    def differ_days(date1,date2):
        a = date1
        b = date2
        return (a - b).days


    day_diff = np.zeros((carData.shape[0],1))
    i = 0
    for i in range(carData.shape[0]):

        if(carData['MonthClaimed'][i]-carData['Month'][i]) < 0:
            year2 = carData['Year'][i] + 1
            month2 = carData['MonthClaimed'][i]
            week2 = carData['WeekOfMonthClaimed'][i]
            day2 = carData['DayOfWeekClaimed'][i]
            year1 = carData['Year'][i]
            month1 = carData['Month'][i]
            week1 = carData['WeekOfMonth'][i]
            day1 = carData['DayOfWeek'][i]
            day_diff[i] = differ_days(get_day(year2,month2,week2,day2),get_day(year1,month1,week1,day1))
        else:
            year2 = carData['Year'][i]
            month2 = carData['MonthClaimed'][i]
            week2 = carData['WeekOfMonthClaimed'][i]
            day2 = carData['DayOfWeekClaimed'][i]
            year1 = carData['Year'][i]
            month1 = carData['Month'][i]
            week1 = carData['WeekOfMonth'][i]
            day1 = carData['DayOfWeek'][i]
            day_diff[i] = differ_days(get_day(year2,month2,week2,day2),get_day(year1,month1,week1,day1))

    #adding column to the existing dataframe
    carData['daysDiff'] = day_diff 
    

    #now drop the original attibutes, like 'Month' column(we don't need anymore)
    carData.drop(['Month'],axis=1,inplace=True)
    carData.drop(['MonthClaimed'],axis=1,inplace=True)
    carData.drop(['DayOfWeek'],axis=1,inplace=True)
    carData.drop(['DayOfWeekClaimed'],axis=1,inplace=True)
    carData.drop(['WeekOfMonth'],inplace=True,axis=1)
    carData.drop(['WeekOfMonthClaimed'],inplace=True,axis=1)
    carData.drop(['VehicleCategory'],axis=1,inplace=True)
    carData.drop(['BasePolicy'],axis=1,inplace=True)
    carData.drop(['Age'],inplace=True,axis=1)
    carData.drop(['PolicyNumber'],inplace=True,axis=1)
    carData.drop(['Year'],inplace=True,axis=1)

    #get the lable of the datasets
    label_Number = LabelEncoder()  #object of lable encoder

    #conver label to Number
    carData['FraudFound'] = label_Number.fit_transform(carData['FraudFound'].astype('str'))

    yLable= carData['FraudFound']

    #drop the lable from the dataset
    carData.drop(['FraudFound'],inplace=True,axis=1)

    #change the data type to numeric
    carData['RepNumber']= pd.to_numeric(carData['RepNumber'])
    carData['Deductible']= pd.to_numeric(carData['Deductible'])
    carData['DriverRating']= pd.to_numeric(carData['DriverRating'])
    #carData.info()

    #select all the attributes of type object
    carObject= carData.select_dtypes(include=['object']).copy()

    #drop the attributes of type object
    carData.drop(carData.select_dtypes(['object']),inplace=True,axis=1)

    #normalization of feature to bring the value in the range [0,1]
    minMaxScale= MinMaxScaler() #minMax scaler
    carData= minMaxScale.fit_transform(carData)

    #converting numpyarry to dataframe
    carData= pd.DataFrame(carData)

    #one hot encoding
    car_copy = carObject.copy()
    car_copy = pd.get_dummies(car_copy,columns=['Make'],prefix=['Make'])
    car_copy = pd.get_dummies(car_copy,columns=['AccidentArea'],prefix=['AccidentArea'])
    car_copy = pd.get_dummies(car_copy,columns=['Sex'],prefix=['Sex'])
    car_copy = pd.get_dummies(car_copy,columns=['MaritalStatus'],prefix=['MartalStatus'])
    car_copy = pd.get_dummies(car_copy,columns=['Fault'],prefix=['Fault'])
    car_copy = pd.get_dummies(car_copy,columns=['PolicyType'],prefix=['PolicyType'])
    car_copy = pd.get_dummies(car_copy,columns=['VehiclePrice'],prefix=['VehiclePrice'])
    car_copy = pd.get_dummies(car_copy,columns=['Days:Policy-Accident'],prefix=['Days:Policy-Accident'])
    car_copy = pd.get_dummies(car_copy,columns=['Days:Policy-Claim'],prefix=['Days:Policy-Claim'])
    car_copy = pd.get_dummies(car_copy,columns=['PastNumberOfClaims'],prefix=['PastNumberOfClaims'])
    car_copy = pd.get_dummies(car_copy,columns=['AgeOfVehicle'],prefix=['AgeOfVehicle'])
    car_copy = pd.get_dummies(car_copy,columns=['AgeOfPolicyHolder'],prefix=['AgeOfPolicyHolder'])
    car_copy = pd.get_dummies(car_copy,columns=['PoliceReportFiled'],prefix=['PoliceReportFiled'])
    car_copy = pd.get_dummies(car_copy,columns=['WitnessPresent'],prefix=['WitnessPresent'])
    car_copy = pd.get_dummies(car_copy,columns=['AgentType'],prefix=['AgentType'])
    car_copy = pd.get_dummies(car_copy,columns=['NumberOfSuppliments'],prefix=['NumberOfSuppliments'])
    car_copy = pd.get_dummies(car_copy,columns=['AddressChange-Claim'],prefix=['AddressChange-Claim'])
    car_copy = pd.get_dummies(car_copy,columns=['NumberOfCars'],prefix=['NumberOfCars'])

    #putt all the attributes together
    car_copy['RepNo']= carData[0]
    car_copy['Deductible']= carData[1]
    car_copy['DriverRating']= carData[2]
    car_copy['DaysDiff']= carData[3]
    car_copy['Lable']= yLable
    
    return car_copy
