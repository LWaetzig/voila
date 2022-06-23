from src.Stock import Stock

stock = Stock("TSLA" , "3y")

df = stock.get_stock_data()

stock.visualize_data(df)

predicted_df = stock.make_prediction()

df = df.append(predicted_df)
stock.visualize_data(df , prediction=True)
stock.evaluate_model(df)
