import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import datasets, linear_model, ensemble, preprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor

from nltk.sentiment.util import *
import random
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import losses, activations

from math import sqrt

data = pd.read_csv(r"RedditNews.csv")

#Removing special characters

x=pd.DataFrame(columns=['Date','News'])
index=0
for index,row in data.iterrows():
    stre=row["News"]
    my_new_string = re.sub('[^ a-zA-Z0-9]', '', stre)
    x.sort_index()
    x.at[index,'Date']=row["Date"]
    x.at[index,'News']=my_new_string
    index=index+1

#Merging same dates
y=pd.DataFrame(columns=['Date','News'])
indx=0
get_news=""
for i in range(0,len(x)-1):
    get_date=x.Date.iloc[i]
    next_date=x.Date.iloc[i+1]
    if(str(get_date)==str(next_date)):
        get_news=get_news+x.News.iloc[i]+" "
    if(str(get_date)!=str(next_date)):
        y.at[indx,'Date']=get_date
        y.at[indx,'News']=get_news
        indx=indx+1
        get_news=" "

read_stock_p=pd.read_csv(r"s&p.csv")

er = pd.to_datetime(y['Date'])

er1 = pd.to_datetime(read_stock_p['Date'])

del y['Date']

del read_stock_p['Date']

y['Date'] = er.values

read_stock_p['Date'] = er1.values

#Adding a "Price" column in our dataframe and fetching the stock price as per the date in our dataframe
y['Prices']=""

indx=0
for i in range (0,len(y)):
    for j in range (0,len(read_stock_p)):
        get_news_date=y.Date.iloc[i]
        get_stock_date=read_stock_p.Date.iloc[j]
        if(str(get_stock_date)==str(get_news_date)):
            y.at[i,'Prices']=int(read_stock_p.Close[j])
            break

#take the mean for the close price and put it in the blank value

mean=0
summ=0
count=0
for i in range(0,len(y)):
    if(y.Prices.iloc[i]!=""):
        summ=summ+int(y.Prices.iloc[i])
        count=count+1
mean=summ/count
for i in range(0,len(y)):
    if(y.Prices.iloc[i]==""):
        y.Prices.iloc[i]=int(mean)


#Making "prices" column as integer so mathematical operations could be performed easily
y['Prices'] = y['Prices'].apply(np.int64)


#Adding 4 new columns in our dataframe so that sentiment analysis could be performed

y["Comp"] = ''
y["Negative"] = ''
y["Neutral"] = ''
y["Positive"] = ''


#for assigning the polarity for each statement. That is how much positive, negative, neutral your statement is
sentiment_i_a = SentimentIntensityAnalyzer()
for indexx, row in y.News.iteritems():
    try:
        sentence_i = unicodedata.normalize('NFKD', y.loc[indexx, 'News'])
        sentence_sentiment = sentiment_i_a.polarity_scores(sentence_i)
        y.at[indexx, 'Comp']= sentence_sentiment['compound']
        y.at[indexx, 'Negative']= sentence_sentiment['neg']
        y.at[indexx, 'Neutral']= sentence_sentiment['neu']
        y.at[indexx, 'Positive']= sentence_sentiment['pos']
    except TypeError:
        print (stocks_dataf.loc[indexx, 'News'])
        print (indexx)

del y['News']

###################################Model Fitting############################################
x1 = y.loc[::-1].reset_index(drop = True)
dl = y.loc[::-1].reset_index(drop = True)

###################################Machine Learning Model####################################

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

# Converting dataset in supervised learning as we need to map stock price of current day to previous day's sentiment
x2 = series_to_supervised(x1, 1, 1)

x2.drop(x2.columns[[0,1,8,9,10,11]], axis=1, inplace=True)

x2 = x2.rename(columns={"var3(t-1)":"Comp","var4(t-1)":"Negative","var5(t-1)":"Neutral","var6(t-1)":"Positive","var1(t)":"Date", "var2(t)":"Prices"})

x = pd.DataFrame()

x['Date'] = x2['Date'].values

x['Prices'] = x2['Prices'].values

x['Comp'] = x2['Comp'].values

x['Negative'] = x2['Negative'].values

x['Neutral'] = x2['Neutral'].values

x['Positive'] = x2['Positive'].values

trainX = x.iloc[0:2553, 2:6]
trainY = x.iloc[0:2553, 1]
testX = x.iloc[ 2554:2941 , 2:6]
testY = x.iloc[ 2554:2941, 1]

#LinearRegression

regr = linear_model.LinearRegression()
regr.fit(trainX, trainY)


y_pred = regr.predict(testX)

mean_absolute_error(testY,y_pred)
print('Coefficients: \n', regr.coef_)

prediction_results_LR = pd.DataFrame()
prediction_results_LR['Actual'] = testY
prediction_results_LR['Predicted'] = y_pred
lr = mean_absolute_error(prediction_results_LR['Actual'],prediction_results_LR['Predicted'])
rmse_lr = sqrt(mean_squared_error(prediction_results_LR['Actual'],prediction_results_LR['Predicted']))

test_prediction_off = pd.DataFrame()
test_prediction_off = prediction_results_LR

test_prediction_off['Predicted'] = test_prediction_off['Predicted'] / 1.2

mean_absolute_error(test_prediction_off['Actual'],test_prediction_off['Predicted'])

#Decision Tree Regressor


regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(trainX, trainY)
regr_2.fit(trainX, trainY)

y_1 = regr_1.predict(testX)
y_2 = regr_2.predict(testX)

prediction_results_DT = pd.DataFrame()
prediction_results_DT['Actual'] = testY
prediction_results_DT = pd.DataFrame()
prediction_results_DT['Actual'] = testY.values
prediction_results_DT['Predicted regressor 2'] = y_2
prediction_results_DT['Predicted regressor 1'] = y_1

dtr = mean_absolute_error(prediction_results_DT['Actual'],prediction_results_DT['Predicted regressor 1'])
rmse_dtr = sqrt(mean_squared_error(prediction_results_DT['Actual'],prediction_results_DT['Predicted regressor 1']))

mean_absolute_error(prediction_results_DT['Actual'],prediction_results_DT['Predicted regressor 2'])


#Gradient Boosting

params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(trainX, trainY)

mse = mean_squared_error(testY, reg.predict(testX))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
#The mean squared error (MSE) on test set: 241420.6419

xg = mean_absolute_error(testY,reg.predict(testX))
rmse_xg = sqrt(mean_squared_error(testY,reg.predict(testX)))
prediction_results_XG = pd.DataFrame()
prediction_results_XG['Actual'] = testY
prediction_results_XG['Predicted'] = reg.predict(testX)
mean_absolute_error(prediction_results_XG['Actual'],prediction_results_XG['Predicted'])



####################################Deep Learning Models#####################################

dl1 = dl.set_index(dl['Date'])
del dl1['Date']

values = dl1.values

values = values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

reframed = series_to_supervised(scaled, 1, 1)

reframed.drop(reframed.columns[[6,7,8,9]], axis=1, inplace=True)

values = reframed.values

train = values[:2553, :]
test = values[2553:, :]

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
print('Test RMSE: %.3f' % rmse_lstm)
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
print(falto)

t = pd.DataFrame(data=np.column_stack((Y_test,y_pred)),columns=['Actual','Predicted'])
mae_ann = mean_absolute_error(Y_test,y_pred)
print('Test MAE: %.3f' % mae_ann)
rmse_ann = sqrt(mean_squared_error(Y_test,y_pred))
print('Test RMSE: %.3f' % rmse_ann)

###########################################Compiling the Results############################
print('Test MAE for linear regression: %.3f' % lr)
print('Test MAE for Decision Tree Regressor: %.3f' % dtr)
print('Test MAE for Extreme Gradient Boosting: %.3f' % xg)
print('Test MAE for LSTM: %.3f' % mae_lstm)
print('Test MAE for ANN: %.3f' % mae_ann)

data = {'Mean Absolute Error':[lr, dtr, xg, mae_lstm, mae_ann],
		'Root Mean Square Error':[rmse_lr, rmse_dtr, rmse_xg, rmse_lstm, rmse_ann]}
df = pd.DataFrame(data, index =['Test for linear regression', 
                                'Test for Decision Tree Regressor', 
                                'Test for Extreme Gradient Boosting', 
                                'Test for LSTM',
								'Test for ANN'])

print(df)