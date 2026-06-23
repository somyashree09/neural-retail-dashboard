import pandas as pd
from prophet import Prophet

# Load your dataset
df = pd.read_csv('global_ecommerce_sales.xls')  

df['Order_Date'] = pd.to_datetime(df['Order_Date'])

daily = df.groupby('Order_Date')['Total_Sales'].sum().reset_index()
daily.columns = ['ds', 'y']

model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.fit(daily)

# Predict next 30 days
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Calculate MAPE on historical data
hist = forecast[forecast['ds'].isin(daily['ds'])]
actual = daily['y'].values
predicted = hist['yhat'].values

mape = (abs(actual - predicted) / actual).mean() * 100
print(f'MAPE: {mape:.2f}%')

out = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
out.columns = ['Date', 'Forecast', 'Lower_Bound', 'Upper_Bound']
out.to_csv('forecast_output.csv', index=False)

print('Saved: forecast_output.csv')