import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('global_ecommerce_sales.xls')
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
snapshot = df['Order_Date'].max()
rfm = df.groupby('Customer_Name').agg(
Recency=('Order_Date', lambda x:(snapshot-x.max()).days),
Frequency=('Order_ID','count'),
Monetary=('Total_Sales','sum')
).reset_index()
scaler = StandardScaler()
scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(scaled)
labels = {0:'Champions',1:'Loyal',2:'At Risk',3:'Lost',4:'Promising'}
rfm['ML_Segment'] = rfm['Cluster'].map(labels)
rfm.to_csv('customer_segments.csv', index=False)
print(rfm['ML_Segment'].value_counts())
print('Saved: customer_segments.csv')