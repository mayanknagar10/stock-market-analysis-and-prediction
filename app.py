from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import cufflinks as cf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.models import load_model
import yahoo_fin.stock_info as si
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import streamlit as st
from datetime import date, timedelta, datetime
from arch import arch_model
from arch.__future__ import reindexing
import yfinance as yf
import os.path

################################################################################

## Functions for calculating SMA, EMA, MACD, RSI
def SMA(data, period = 100, column = 'Adj Close'):
        return data[column].rolling(window=period).mean()

def EMA(data, period = 20, column = 'Adj Close'):
        return data[column].ewm(span=period, adjust = False).mean()

def MACD(data, period_long = 26, period_short = 12, period_signal = 9, column = 'Adj Close'):
        shortEMA = EMA(data, period_short, column=column)
        longEMA = EMA(data, period_long, column=column)
        data['MACD'] = shortEMA - longEMA
        data['Signal_Line'] = EMA(data, period_signal, column = 'MACD')
        return data

def RSI(data, period = 14, column = 'Adj Close'):
        delta = data[column].diff(1)
        delta = delta[1:]
        up = delta.copy()
        down = delta.copy()
        up[up<0] = 0
        down[down>0] = 0
        data['up'] = up
        data['down'] = down
        avg_gain = SMA(data, period, column = 'up')
        avg_loss = abs(SMA(data, period, column = 'down'))
        RS = avg_gain/avg_loss
        RSI = 100.0 - (100.0/(1.0+RS))
        data['RSI'] = RSI
        return data

def BB(data):
        data['TP'] = (data['Adj Close'] + data['Low'] + data['High'])/3
        data['std'] = data['TP'].rolling(20).std(ddof=0)
        data['MA-TP'] = data['TP'].rolling(20).mean()
        data['BOLU'] = data['MA-TP'] + 2*data['std']
        data['BOLD'] = data['MA-TP'] - 2*data['std']
        return data

## Function for plotting Stock Prices, Volume, Indicators & Returns
def get_stock_price_fig(df,v2,v3):

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_width=[0.1,0.2,0.1, 0.3],subplot_titles=("", "", v2, v3 + ' %'))

        fig.add_trace(go.Candlestick(
                        x=df['Date'],
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Adj Close'],showlegend = False, name = 'Price'),row=1,col=1)

        fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'],opacity=0.5,showlegend = False, name = 'Volume'),
        row = 2, col= 1)

        # Indicators
        if v2=='RSI':
                fig.add_trace(go.Scatter(x = df['Date'], y=df['RSI'], mode="lines", name = 'RSI',
                marker=dict(color='rgb(31, 119, 180)'), showlegend = False),row = 3, col= 1)
                fig.layout.xaxis.showgrid=False
        elif v2=='SMA':
                fig.add_trace(go.Scatter(x = df['Date'], y=df['SMA_50'], mode="lines", name = 'SMA_50', 
                showlegend = False, marker=dict(color='rgb(31, 119, 180)')),row = 3, col= 1)
                fig.add_trace(go.Scatter(x = df['Date'], y=df['SMA_200'], mode="lines", name = 'SMA_200', 
                showlegend = False, marker=dict(color='#ff3333')),row = 3, col= 1)
                fig.layout.xaxis.showgrid=False
        elif v2=='EMA':
                fig.add_trace(go.Scatter(x = df['Date'], y=df['EMA'], mode="lines", name = 'EMA', 
                showlegend = False, marker=dict(color='rgb(31, 119, 180)')),row = 3, col= 1)
                fig.layout.xaxis.showgrid=False
        elif v2=='MACD':
                fig.add_trace(go.Scatter(x = df['Date'], y=df['MACD'], mode="lines",name = 'MACD', 
                showlegend = False, marker=dict(color='rgb(31, 119, 180)')),row = 3, col= 1)
                fig.add_trace(go.Scatter(x = df['Date'], y=df['Signal_Line'], mode="lines",name='Signal_Line', 
                showlegend = False, marker=dict(color='#ff3333')),row = 3, col= 1)
                fig.layout.xaxis.showgrid=False
        elif v2=='Bollinger Bands':
                fig.add_trace(go.Scatter(x = df['Date'], y=df['Adj Close'], mode="lines",
                line=dict(color='rgb(31, 119, 180)'),name = 'Close',showlegend = False),row = 3, col= 1) 
                fig.add_trace(go.Scatter(x = df['Date'], y=df['BOLU'],mode="lines", line=dict(width=0.5), 
                marker=dict(color="#89BCFD"),showlegend=False,name = 'Upper Band'),row = 3, col= 1)
                fig.add_trace(go.Scatter(x = df['Date'], y=df['BOLD'], mode="lines",line=dict(width=0.5),
                marker=dict(color="#89BCFD"),showlegend=False,fillcolor='rgba(56, 224, 56, 0.5)',fill='tonexty',name = 'Lower Band'),row = 3, col= 1)
                fig.layout.xaxis.showgrid=False        

        # Returns
        if v3=="Daily Returns":
                rets = df['Adj Close']/df['Adj Close'].shift(1) - 1
                fig.add_trace(go.Scatter(x = df['Date'], y=rets, mode="lines", showlegend = False, name = 'Daily Return', line=dict(color='#FF4136')),
                row = 4, col= 1,)
                fig.layout.xaxis.showgrid=False
        elif v3=="Cumulative Returns":
                rets = df['Adj Close']/df['Adj Close'].shift(1) - 1
                cum_rets = (rets + 1).cumprod()
                fig.add_trace(go.Scatter(x = df['Date'], y=cum_rets, mode="lines", showlegend = False, name = 'Cumulative Returns', line=dict(color='#FF4136')),
                row = 4, col=1)
                fig.layout.xaxis.showgrid=False

        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.update_layout(margin=dict(b=0,t=0,l=0,r=0),plot_bgcolor='#ebf3ff',width=500, height=600, 
                        xaxis_showticklabels=True, xaxis4_showticklabels=False, xaxis3_showgrid = False, xaxis4_showgrid = False)
        fig.layout.xaxis.showgrid=False
        return fig

## Function for calculating Alpha & Beta ratio







##########################################################################################################################

st.markdown('''
# Stock Market Analysis and Prediction
''')
st.write('---')


st.header('INFORMATION')

st.sidebar.subheader('Query parameters')
period = pd.read_csv('https://raw.githubusercontent.com/mayanknagar10/stock_market_analysis_and_prediction/main/period.csv')
time = st.sidebar.selectbox("Time Period", period)

indicators_period = pd.read_csv('https://raw.githubusercontent.com/mayanknagar10/stock_market_analysis_and_prediction/main/indicators_period.csv')
time_period_input = st.sidebar.selectbox("Time Period for indicators",indicators_period.indicators_time)
if time_period_input=='6m':
    time_period = 126
elif time_period_input=='1y':
    time_period = 252
elif time_period_input=='3y':
    time_period = 756
elif time_period_input=='5y':
    time_period = 1800
# st.sidebar.write('Note : ')

# start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
# end_date = st.sidebar.date_input("End date", datetime.date(2021, 1, 31))
ticker_list = pd.read_csv('https://raw.githubusercontent.com/mayanknagar10/stock_market_analysis_and_prediction/main/company_data.csv')
user_input = st.sidebar.selectbox('Enter Stock Ticker', ticker_list)

st.sidebar.write('For other Ticker or Company refer a yahoo finance website : https://finance.yahoo.com/')

indicators = st.sidebar.radio("Indicators", ('SMA', 'EMA', 'MACD', 'RSI'))

returns = st.sidebar.radio("Returns", ('Daily Returns', 'Cumulative Returns'))

df = yf.download(user_input,period=time,interval='1d')
df.head()


information = yf.Ticker(user_input)  

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
df['Date'] = df.index
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

# st.subheader('Analysis and Some Charts')

st.subheader("Stock Price Chart : ")

fig = plt.figure(figsize = (10,6))
df['Open'].plot(label = 'Open Price')
df['Close'].plot(label = 'Close Price')
df['High'].plot(label = 'High Price')
df['Low'].plot(label = 'Low Price')
plt.legend()
plt.ylabel('Stock Price')
st.plotly_chart(fig)

st.subheader("Volume Traded Chart : ")

fig = plt.figure(figsize = (10,6))
df['Volume'].plot()
st.plotly_chart(fig)


###########################################

st.subheader('Indicators and Returns')



MACD(df)
RSI(df)
BB(df)
df['SMA_50'] = SMA(df, 50)
df['SMA_200'] = SMA(df, 200)
df['EMA'] = EMA(df)
fig = get_stock_price_fig(df.tail(time_period),indicators,returns)
st.plotly_chart(fig)


# with st.expander("See explanation about this indicators :"):
#      st.write("""
#          The chart above shows some numbers I picked for you.
#          I rolled actual dice for these, so they're *guaranteed* to
#          be random.
#      """)
##########################################


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

