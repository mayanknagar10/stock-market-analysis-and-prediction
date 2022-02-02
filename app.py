from time import time
from matplotlib.style import use
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yahooFin  
import cufflinks as cf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.models import load_model
import yahoo_fin.stock_info as si
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import yfinance as yf

st.markdown('''
# Stock Market Analysis and Prediction
''')
st.write('---')


st.header('INFORMATION')

st.sidebar.subheader('Query parameters')
time = st.sidebar.text_input("Period",'5y')
# start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
# end_date = st.sidebar.date_input("End date", datetime.date(2021, 1, 31))

user_input = st.sidebar.text_input('Enter Stock Ticker', 'TATAMOTORS.NS')

df = yf.download(user_input,period=time,interval='1d')
df.head()


information = yahooFin.Ticker(user_input)  

string_logo = '<img src=%s>' % information.info['logo_url']
st.markdown(string_logo, unsafe_allow_html=True)

string_name = information.info['longName']
st.header('**%s**' % string_name)

string_summary = information.info['longBusinessSummary']
st.info(string_summary)



df1 = df.reset_index()
df1 = df1.drop(['Date', 'Adj Close'], axis = 1)
df1.head()

st.subheader('Candlestick chart')

candlestick = go.Candlestick(x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'])

fig = go.Figure(data=[candlestick])
st.plotly_chart(fig,use_container_width=True)


st.subheader('Data')

st.write(df.tail(2))

st.subheader('Fundamentals')

data1 = si.get_quote_table(user_input)

FiftyTwo_week_range = data1["52 Week Range"]
day_range = data1["Day's Range"]
avg_volume = data1["Avg. Volume"]
eps = data1["EPS (TTM)"]
marketcap = data1["Market Cap"]
pe = data1["PE Ratio (TTM)"]
volume = data1["Volume"]
quote_price = data1["Quote Price"]

st.write("52 Week Range : ", FiftyTwo_week_range)
st.write("Day's Range : ", day_range)
st.write("Average Volume : ", avg_volume)
st.write("EPS : ", eps)
st.write("Market Cap : ", marketcap)
st.write("PE Ratio : ", pe)
st.write("Volume : ", volume)
st.write("Quote Price : ", quote_price)

company = yf.Ticker(user_input)
st.write("Major Holders : ", company.major_holders)
st.write("Institutional Holders : ", company.institutional_holders)

st.subheader('Analysis and Some Charts')

st.write("Stock Price Chart : ")

fig = plt.figure(figsize = (12,6))
df['Open'].plot(label = 'Open Price')
df['Close'].plot(label = 'Close Price')
df['High'].plot(label = 'High Price')
df['Low'].plot(label = 'Low Price')
plt.legend()
plt.title('Company Stock Price')
plt.ylabel('Stock Price')
st.pyplot(fig)

st.write("Volume Traded Chart : ")

fig = plt.figure(figsize = (12,6))
df['Volume'].plot(figsize=(17,5))
plt.title('Volume Traded')
st.pyplot(fig)

st.write("Daily Returns Chart : ")

daily_returns = df['Adj Close'].pct_change()
fig = plt.figure(figsize = (12,6))
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.plot(daily_returns)
ax1.set_xlabel("Date")
ax1.set_ylabel("Percent")
ax1.set_title("daily returns data")
st.pyplot(fig)

st.write("First Quant Figure : ")

qf=cf.QuantFig(df,title='First Quant Figure',legend='top',name='GS')
qf.add_bollinger_bands()
fig = qf.iplot(asFigure=True)
st.plotly_chart(fig)

st.subheader('Result and Prediction')

close = df[['Close']]
ds = close.values
normalizer = MinMaxScaler(feature_range=(0,1))
ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))

st.write('Closing Price vs Time chart with 100MA : ')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.write('Closing Price vs Time chart with 100MA & 200 : ')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

train_size = int(len(ds_scaled)*0.70)
test_size = len(ds_scaled) - train_size

ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled),:1]

def create_ds(dataset,step):
    Xtrain, Ytrain = [], []
    for i in range(len(dataset)-step-1):
        a = dataset[i:(i+step), 0]
        Xtrain.append(a)
        Ytrain.append(dataset[i + step, 0])
    return np.array(Xtrain), np.array(Ytrain)

time_stamp = 100
X_train, y_train = create_ds(ds_train,time_stamp)
X_test, y_test = create_ds(ds_test,time_stamp)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

model = load_model('keras_model.h5')

# loss = model.history.history['loss']
# plt.plot(loss)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = normalizer.inverse_transform(train_predict)
test_predict = normalizer.inverse_transform(test_predict)

plt.plot(normalizer.inverse_transform(ds_scaled))
plt.plot(train_predict)
plt.plot(test_predict)

test = np.vstack((train_predict,test_predict))

plt.plot(normalizer.inverse_transform(ds_scaled))
plt.plot(test)


fut_inp = ds_test[(len(ds_test)-100):]

fut_inp = fut_inp.reshape(1,-1)

tmp_inp = list(fut_inp)

tmp_inp = tmp_inp[0].tolist()

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(tmp_inp)>100):
        fut_inp = np.array(tmp_inp[1:])
        fut_inp =fut_inp.reshape(1,-1)
        fut_inp = fut_inp.reshape((1, n_steps, 1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        tmp_inp = tmp_inp[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        fut_inp = fut_inp.reshape((1, n_steps,1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i=i+1
    

plot_new=np.arange(1,101)
plot_pred=np.arange(101,131)


#result
st.write('Result : ')
fig = plt.figure(figsize = (12,6))
plt.plot(plot_new, normalizer.inverse_transform(ds_scaled[(len(ds_scaled)-100):]))
plt.plot(plot_pred, normalizer.inverse_transform(lst_output))
st.pyplot(fig)

ds_new = ds_scaled.tolist()

ds_new.extend(lst_output)

plt.plot(ds_new[(len(ds_new)-50):])

final_graph = normalizer.inverse_transform(ds_new).tolist()

st.write('Prediction : ')
fig = plt.figure(figsize = (12,6))
plt.plot(final_graph,)
plt.ylabel("Price")
plt.xlabel("Time")
plt.title("{0} prediction of next month close".format(user_input))
plt.axhline(y=final_graph[len(final_graph)-1], color = 'red', linestyle = ':', label = 'NEXT 30D: {0}'.format(round(float(*final_graph[len(final_graph)-1]),2)))
plt.legend()
st.pyplot(fig)

price = round(float(*final_graph[len(final_graph)-1]),2)
st.write("Next 30D price : ", price)


