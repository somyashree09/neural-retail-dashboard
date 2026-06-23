import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# Load your dataset (change filename if needed)
df = pd.read_csv('global_ecommerce_sales.xls')
# Convert date and build RFM features per customer
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
snapshot = df['Order_Date'].max()
rfm = df.groupby('Customer_Name').agg(
Recency=('Order_Date', lambda x: (snapshot-x.max()).days),
Frequency=('Order_ID', 'count'),
Monetary=('Total_Sales', 'sum')
).reset_index()
# Label: churned = inactive >60 days
rfm['Churned'] = (rfm['Recency'] > 60).astype(int)
X = rfm[['Recency','Frequency','Monetary']]
y = rfm['Churned']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)
# Print AUC-ROC score (put this number in your report!)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
print(f'AUC-ROC Score: {auc:.4f}') # Note this number!
# Save churn scores for every customer
rfm['Churn_Probability'] = model.predict_proba(X)[:,1].round(4)
rfm['Churn_Risk'] = rfm['Churn_Probability'].apply(
lambda x: 'High' if x>0.7 else ('Medium' if x>0.4 else 'Low'))
rfm.to_csv('churn_predictions.csv', index=False)
print('Saved: churn_predictions.csv')