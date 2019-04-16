
# coding: utf-8

# In[102]:

import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler
import calendar
import datetime
from datetime import datetime
get_ipython().magic('matplotlib inline')


# In[105]:

#read the data.
car_df = pd.read_csv('../../Data/newCardata.csv')

print(car_df.shape)
car_df.head()


# In[106]:

#replace map
replace_Month = {'Month':{'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
                          'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}}

replace_MonthClaimed = {'MonthClaimed':{'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
                          'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}}

car_df.replace(replace_Month,inplace=True)
car_df.replace(replace_MonthClaimed,inplace=True)

car_df.head()


# In[ ]:




# In[107]:

#function to calculate the no of days passed between the accident and the claims.
# Reporting Gap:

def get_day(year,month,weekOfMonth,dayOfWeek):
    count = 0
    c = calendar.TextCalendar(calendar.SUNDAY)
    l = []
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


# In[108]:


#checking the instance of the data
if(car_df['MonthClaimed'][0]-car_df['Month'][0]) < 0:
    date1= get_day(1994,12,5,'Wednesday')
    date2= get_day(1995,1,1,'Tuesday')
    print(date1,date2)


# In[109]:

day_diff = np.zeros((car_df.shape[0],1))
i = 0
for i in range(car_df.shape[0]):
    
    if(car_df['MonthClaimed'][i]-car_df['Month'][i]) < 0:
        year2 = car_df['Year'][i] + 1
        month2 = car_df['MonthClaimed'][i]
        week2 = car_df['WeekOfMonthClaimed'][i]
        day2 = car_df['DayOfWeekClaimed'][i]
        year1 = car_df['Year'][i]
        month1 = car_df['Month'][i]
        week1 = car_df['WeekOfMonth'][i]
        day1 = car_df['DayOfWeek'][i]
        day_diff[i] = differ_days(get_day(year2,month2,week2,day2),get_day(year1,month1,week1,day1))
    else:
        year2 = car_df['Year'][i]
        month2 = car_df['MonthClaimed'][i]
        week2 = car_df['WeekOfMonthClaimed'][i]
        day2 = car_df['DayOfWeekClaimed'][i]
        year1 = car_df['Year'][i]
        month1 = car_df['Month'][i]
        week1 = car_df['WeekOfMonth'][i]
        day1 = car_df['DayOfWeek'][i]
        day_diff[i] = differ_days(get_day(year2,month2,week2,day2),get_day(year1,month1,week1,day1))
        


# In[110]:

#adding column to the existing dataframe
car_df['daysDiff'] = day_diff 
car_df.head(3)


# In[111]:

#now drop the original attibutes, like 'AccidentArea' column(we don't need anymore)
car_df.drop(['Month'],axis=1,inplace=True)
car_df.drop(['MonthClaimed'],axis=1,inplace=True)
car_df.drop(['DayOfWeek'],axis=1,inplace=True)
car_df.drop(['DayOfWeekClaimed'],axis=1,inplace=True)
car_df.drop(['WeekOfMonth'],inplace=True,axis=1)
car_df.drop(['WeekOfMonthClaimed'],inplace=True,axis=1)
car_df.drop(['VehicleCategory'],axis=1,inplace=True)
car_df.drop(['BasePolicy'],axis=1,inplace=True)
car_df.drop(['Age'],inplace=True,axis=1)
car_df.drop(['PolicyNumber'],inplace=True,axis=1)
car_df.drop(['Year'],inplace=True,axis=1)


# In[112]:

car_df.head()


# In[113]:

#get the lable of the datasets


label_Number = LabelEncoder()  #object of lable encoder

#conver label to Number
car_df['FraudFound'] = label_Number.fit_transform(car_df['FraudFound'].astype('str'))

yLabel = car_df['FraudFound']

#drop the lable from the dataset
car_df.drop(['FraudFound'],inplace=True,axis=1)

print(yLabel.shape)


# In[114]:

car_df.head()


# In[115]:


#select all the attributes of type object
carObject= car_df.select_dtypes(include=['object']).copy()

#drop the attributes of type object
car_df.drop(car_df.select_dtypes(['object']),inplace=True,axis=1)

car_df.head()
#print(type(carDate))


# In[124]:

#normalization of feature to bring the value in the range [0,1]

minMaxScale= MinMaxScaler() #minMax scaler
car_df= minMaxScale.fit_transform(car_df)

#converting numpyarry to dataframe
car_df = pd.DataFrame(car_df)
car_df.head()




# In[ ]:




# In[123]:

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

car_copy.head(3)


# In[129]:

car_copy['RepNo']= car_df[0]
car_copy['Deductible']= car_df[2]
car_copy['DriverRating']= car_df[2]
car_copy['DaysDiff']= car_df[3]

car_copy.head(3)




# In[130]:

#saving the final data after preprocessing:

#saving dataframe to the csv format
car_copy.to_csv('carDatafinal.csv',sep=' ',encoding='utf-8')


# In[ ]:



