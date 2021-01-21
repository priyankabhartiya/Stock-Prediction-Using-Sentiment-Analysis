# Stock-Prediction-Using-Sentiment-Analysis
Unlike the conventional stock market prediction systems, novel approach combines the sentiments of common people through the news feeds and stock price data to predict the behaviour of stock market.

In this project, we have used various natural language processing techniques and machine learning algorithms to predict stock price actual value using sentiment analysis from python.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

# Prerequistes

1) This project is developed at python 3.7
2) Second and easier option is to download anaconda and use its anaconda prompt to run the commands. To install anaconda check this url https://www.anaconda.com/download/
3) You will also need to download and install below 5 packages after you install either python or anaconda from the steps above
Sklearn (sciket-Learn), numpy, Keras, NLTK, Matplotlib

# Dataset Employed

Two types of datasets are employed for sentiment analysis:
1) ReditNews Dataset: It incorporates top trending news of the day from the period of 2008 to 2016. Dataset contains attribute date and Headlines.
2) Top 25 News Dataset: It incorporates the top 25 headlines of the day for the time scale of 2008 to 2016. Dataset contains 26 attributes, 25 for each headline and 1 for Date.
3) S&P 500 Index: It contains Date, Open, Close, High, Low, and Adj. Close fields, which contains stock price values for each day.

# Install and Run the code

The first step would be to clone this repo in a folder in your local machine. To do that you need to run following command in command prompt or in git bash

1) Open Anaconda Prompt
###### For Reddit News Dataset follow commands

```
cd C:/your cloned project folder path goes here/
cd Reddit News
python ml_models_Reddit.py
```
###### For Top 25 News Headline Dataset
```
cd C:/your cloned project folder path goes here/
cd "Top 25 Headlines"
python ml_models_DJIA.py
```    
2) If you have chosen to install python (and did not set up PATH variable for it) then follow the below instructions:

Open command prompt and change the directory to project directory by running below command.

`cd C:/your cloned project folder path goes here/`

Locate python.exe in your machine.

Once you locate the python.exe path, you need to write whole path of it and then entire path of project folder with prediction.py at the end. For example if your python.exe is located at c:/Python36/python.exe and project folder is at c:/users/user_name/desktop/Stock_Price_Forecasting_using_Sentiment_Analysis/, then your command to run program will be as below:

```
C:/Python37/python.exe C:/users/user_name/desktop/Stock_Price_Forecasting_using_Sentiment_Analysis/ml_models_Reddit
```

or

```
C:/Python37/python.exe C:/users/user_name/desktop/Stock_Price_Forecasting_using_Sentiment_Analysis/ml_models_DJIA
```

### Sentiment Analysis will take few minutes to execute, so all in all programs for both dataset will take few minutes for excution and once the execution is complete final error matrix will be promted on the screen as a final result.
