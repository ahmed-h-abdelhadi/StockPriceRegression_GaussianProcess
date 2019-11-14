from talib import abstract
import pandas as pd
import numpy as np


# instantiate TA-LIB functions
CCI = abstract.Function('cci')
MACD = abstract.Function('macd')
SMA = abstract.Function('sma')
RSI = abstract.Function('rsi')


# import original dataset
original_data = pd.read_excel(open('./data/DAX_data.xlsx', 'rb'))

# inputs should be provided in "OHLCV" format for the TA-LIB functions
inputs = original_data[['Open', 'High', 'Low', 'Close', 'Volume']]
inputs.columns = inputs.columns.str.lower()

# CCI, Commodity Channel Index, timeperiod = the past 14 observations
cci_output = CCI(inputs, timeperiod=14)

#MACD, Moving Average Convergence/Diverence: fast period = 12, slow
#period = 26, signal period = 9.
macd_output = MACD(inputs, fastperiod=12, slowperiod=26, signalperiod=9)

# Use the generated 'macdhist' data
macd_output = macd_output['macdhist']

# SMA, Simple Moving Average: timeperiod = the past 200 observations.
sma_output = SMA(inputs, timeperiod=200)

# RSI, Relative Strength Index: timeperiod = the past 14 observations.
rsi_output = RSI(inputs, timeperiod=14)

#final output: 5 features (Closing price + the 4 technical indicators) + drop NaN rows
final_output = pd.concat([inputs['close'], cci_output, macd_output, sma_output, rsi_output], axis = 1).dropna()

# save the final output as csv
final_output.to_csv("./data/final_output.csv", sep='\t', encoding='utf-8', index=False, header = False)

# DONE! :)
print("DONE")

