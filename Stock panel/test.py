from pandas_datareader import data as pdr
import yfinance as yf


yf.pdr_override() 
data = pdr.get_data_yahoo("NFLX", start="2022-05-01", end="2022-05-20")
modified_dataframe = data
modified_dataframe.to_excel('daten ntflx 02-19.xlsx')