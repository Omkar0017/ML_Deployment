import numpy as np
import pandas as pd


initialDf = pd.read_csv('Clean_Dataset.csv')
initialDf = initialDf.drop(['Unnamed: 0','flight'],axis = 1)

x = initialDf.iloc[:,:-1]
y = initialDf.iloc[:,-1]

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
x1 = pd.get_dummies(initialDf['airline'])
x2 = pd.get_dummies(initialDf['source_city'])
x3 = pd.get_dummies(initialDf['destination_city'])


from sklearn.preprocessing import OneHotEncoder,LabelEncoder, OrdinalEncoder

OE1 = OneHotEncoder(handle_unknown='ignore')
airLine = OE1.fit_transform(initialDf[['airline']])
airLineDf = pd.DataFrame(airLine.toarray())
airLineDf.columns = OE1.get_feature_names()


OE2 = OneHotEncoder(handle_unknown='ignore')
source = OE2.fit_transform(initialDf[['source_city']])
sourceDf = pd.DataFrame(source.toarray())
sourceDf.columns = OE2.get_feature_names()


df = initialDf[['duration','days_left']]

finalDf = df.join(airLineDf).join(sourceDf)

OE3 = OneHotEncoder(handle_unknown='ignore')
destination = OE3.fit_transform(initialDf[['destination_city']])
destinationDf = pd.DataFrame(destination.toarray())

destinationDf.columns = ['d0_Bangalore', 'd0_Chennai', 'd0_Delhi', 'd0_Hyderabad',
       'd0_Kolkata', 'd0_Mumbai']


finalDf = finalDf.join(destinationDf)

LE1 = OrdinalEncoder(categories=[['Evening', 'Early_Morning', 'Morning', 'Afternoon', 'Night',
       'Late_Night']])
encoder=LE1.fit_transform(initialDf[['departure_time']])
dTimeDf = pd.DataFrame(encoder)
dTimeDf.columns=['departure_time']


LE2 = OrdinalEncoder(categories=[['zero', 'one', 'two_or_more']])
encoder=LE2.fit_transform(initialDf[['stops']])
stopsDf = pd.DataFrame(encoder)
stopsDf.columns=['stops']

LE3 = OrdinalEncoder(categories=[['Evening', 'Early_Morning', 'Morning', 'Afternoon', 'Night',
       'Late_Night']])
encoder=LE3.fit_transform(initialDf[['arrival_time']])
aTimeDf = pd.DataFrame(encoder)
aTimeDf.columns=['arrival_time']

LE4 = OrdinalEncoder(categories=[['Economy', 'Business']])
encoder=LE4.fit_transform(initialDf[['class']])
classDf = pd.DataFrame(encoder)
classDf.columns=['class']

finalDf = finalDf.join(dTimeDf).join(aTimeDf).join(stopsDf).join(classDf)

from sklearn.preprocessing import StandardScaler


sc = StandardScaler()
finalDf = sc.fit_transform(finalDf)


from sklearn.tree import DecisionTreeRegressor
Dt = DecisionTreeRegressor(random_state= 0)  
Dt.fit(finalDf,y)


import pickle
pickle.dump(Dt,open('Dt.pkl','wb'))

model = pickle.load(open('Dt.pkl','rb'))