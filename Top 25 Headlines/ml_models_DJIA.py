import pandas as pd
import matplotlib.pyplot as  plt
import csv
import random
import numpy as np

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *

from sklearn import datasets, linear_model, ensemble, preprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import losses, activations

import unicodedata
from math import sqrt


df=pd.read_csv(r"Combined_News_DJIA.csv")

#Removing Special Characters
data=df.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index

for index in new_Index:
    data[index]=data[index].str.lower()

#Merging Headlines
headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))


x = pd.DataFrame()

x['Date'] = df['Date']

x['Headlines'] = headlines

#Adding a "Price" column in our dataframe and fetching the stock price as per the date in our dataframe
x['Prices']=""
read_stock_p=pd.read_csv(r"s&p.csv")
indx=0
for i in range (0,len(x)):
    for j in range (0,len(read_stock_p)):
        get_news_date=x.Date.iloc[i]
        get_stock_date=read_stock_p.Date.iloc[j]
        if(str(get_stock_date)==str(get_news_date)):
            x.at[i,'Prices']=int(read_stock_p.Close[j])
            break

#take the mean for the close price and put it in the blank value
mean=0
summ=0
count=0
for i in range(0,len(x)):
    if(x.Prices.iloc[i]!=""):
        summ=summ+int(x.Prices.iloc[i])
        count=count+1
mean=summ/count
for i in range(0,len(x)):
    if(x.Prices.iloc[i]==""):
        x.Prices.iloc[i]=int(mean)

#Making "prices" column as integer so mathematical operations could be performed easily
x['Prices'] = x['Prices'].apply(np.float64)


#Adding 4 new columns in our dataframe so that sentiment analysis could be performed
x["Comp"] = ''
x["Negative"] = ''
x["Neutral"] = ''
x["Positive"] = ''


#for assigning the polarity for each statement. That is how much positive, negative, neutral your statement is
sentiment_i_a = SentimentIntensityAnalyzer()
for indexx, row in x.Headlines.iteritems():
    try:
        sentence_i = unicodedata.normalize('NFKD', x.loc[indexx, 'Headlines'])
        sentence_sentiment = sentiment_i_a.polarity_scores(sentence_i)
        x.at[indexx, 'Comp']= sentence_sentiment['compound']
        x.at[indexx, 'Negative']= sentence_sentiment['neg']
        x.at[indexx, 'Neutral']= sentence_sentiment['neu']
        x.at[indexx, 'Positive']= sentence_sentiment['pos']
    except TypeError:
        print (stocks_dataf.loc[indexx, 'Headlines'])
        print (indexx)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

del x['Headlines']
dl = x

##################################################Machine Learning Models#####################################

# Converting dataset in supervised learning as we need to map stock price of current day to previous day's sentiment
x1 = series_to_supervised(x, 1, 1)

x1.drop(x1.columns[[0,1,8,9,10,11]], axis=1, inplace=True)

x1 = x1.rename(columns={"var3(t-1)":"Comp","var4(t-1)":"Negative","var5(t-1)":"Neutral","var6(t-1)":"Positive","var1(t)":"Date", "var2(t)":"Prices"})

x = pd.DataFrame()

x['Date'] = x1['Date'].values

x['Prices'] = x1['Prices'].values

x['Comp'] = x1['Comp'].values

x['Negative'] = x1['Negative'].values

x['Neutral'] = x1['Neutral'].values

x['Positive'] = x1['Positive'].values

trainX = x.iloc[0:1622, 3:7]
trainY = x.iloc[0:1622, 2]
testX = x.iloc[ 1622:1989 , 3:7]
testY = x.iloc[ 1622:1989, 2]

#Linear Regression
regr = linear_model.LinearRegression()
regr.fit(trainX, trainY)

y_pred = regr.predict(testX)

prediction_results = pd.DataFrame()
prediction_results['Actual'] = testY
prediction_results['Predicted'] = y_pred

lr_mae = mean_absolute_error(prediction_results['Actual'],prediction_results['Predicted'])
rmse_lr = sqrt(mean_squared_error(prediction_results['Actual'],prediction_results['Predicted']))

#Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(trainX, trainY)
regr_2.fit(trainX, trainY)

y_1 = regr_1.predict(testX)
y_2 = regr_2.predict(testX)

prediction_results = pd.DataFrame()
prediction_results['Actual'] = testY
prediction_results = pd.DataFrame()
prediction_results['Actual'] = testY.values
prediction_results['Predicted regressor 2'] = y_2
prediction_results['Predicted regressor 1'] = y_1

dtr_mae = mean_absolute_error(prediction_results['Actual'],prediction_results['Predicted regressor 1'])
rmse_dtr = sqrt(mean_squared_error(prediction_results['Actual'],prediction_results['Predicted regressor 1']))

mean_absolute_error(prediction_results['Actual'],prediction_results['Predicted regressor 2'])

#### Extreme Gradient Boosting
params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(trainX, trainY)

mse = mean_squared_error(testY, reg.predict(testX))
xg = mean_absolute_error(testY,reg.predict(testX))
rmse_xg = sqrt(mean_squared_error(testY,reg.predict(testX)))

#Support Vector Regression

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(trainX, trainY)

y_pred = regr.predict(testX)

prediction_results = pd.DataFrame()
prediction_results['Actual'] = testY
prediction_results['Predicted'] = y_pred

mae_svr = mean_absolute_error(prediction_results['Actual'],prediction_results['Predicted'])
rmse_svr = sqrt(mean_squared_error(testY,reg.predict(testX)))

##############################################Deep Learning#######################################

dl1 = dl.set_index(dl['Date'])
del dl1['Date']

values = dl1.values

values = values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

reframed = series_to_supervised(scaled, 1, 1)

reframed.drop(reframed.columns[[6,7,8,9]], axis=1, inplace=True)

values = reframed.values

train = values[:1622, :]
test = values[1622:, :]

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

h = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)


y_pred = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

inv_y_pred = np.concatenate((y_pred, test_X[:, 1:]), axis=1)
inv_y_pred = scaler.inverse_transform(inv_y_pred)
inv_y_pred = inv_y_pred[:,0]

test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

rmse_lstm = sqrt(mean_squared_error(inv_y, inv_y_pred))
mae_lstm = mean_absolute_error(inv_y, inv_y_pred)

###########################ANN##########################


X=dl1[['Comp','Negative','Neutral','Positive']]

Y=dl1[['Prices']]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)


min_max_scalar=preprocessing.MinMaxScaler()
X_train=min_max_scalar.fit_transform(X_train)
X_test=min_max_scalar.fit_transform(X_test)
Y_train=min_max_scalar.fit_transform(Y_train)
Y_test=min_max_scalar.fit_transform(Y_test)

model=Sequential()
model.add(Dense(4,activation=activations.sigmoid,input_shape=(4,)))
model.add(Dense(3,activation=activations.sigmoid))
model.add(Dense(2,activation=activations.sigmoid))
model.add(Dense(1,activation=activations.sigmoid))

model.compile(optimizer='adam',loss=losses.mean_absolute_error)

model.fit(X_train,Y_train,epochs=500)
y_pred=model.predict(X_test)
falto=r2_score(Y_test,y_pred)


t = pd.DataFrame(data=np.column_stack((Y_test,y_pred)),columns=['Actual','Predicted'])
mae_ann = mean_absolute_error(Y_test,y_pred)
rmse_ann = sqrt(mean_squared_error(Y_test,y_pred))

###########################################Compiling the Results############################

data = {'Mean Absolute Error':[lr_mae, dtr_mae, xg, mae_svr, mae_lstm, mae_ann],
		'Root Mean Square Error':[rmse_lr, rmse_dtr, rmse_xg, rmse_svr, rmse_lstm, rmse_ann]}
df = pd.DataFrame(data, index =['Test for linear regression', 
                                'Test for Decision Tree Regressor', 
                                'Test for Extreme Gradient Boosting', 
								'Test for Support Vector Regressor',
                                'Test for LSTM',
								'Test for ANN'])

print (df)