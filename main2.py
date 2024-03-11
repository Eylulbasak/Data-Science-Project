import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

warnings.filterwarnings('ignore')

df = pd.read_csv("dataset.csv", delimiter=";")
data = df.copy()

#print(df.info())
#print(df.shape)

#variables by type
num_columns = ["account_amount_added_12_24m", "account_days_in_dc_12_24m", "account_days_in_rem_12_24m",
               "account_days_in_term_12_24m", "account_incoming_debt_vs_paid_0_24m", "age", "avg_payment_span_0_12m",
               "avg_payment_span_0_3m", "max_paid_inv_0_12m", "max_paid_inv_0_24m", "num_active_div_by_paid_inv_0_12m",
               "num_active_inv", "num_arch_dc_0_12m", "num_arch_dc_12_24m", "num_arch_ok_0_12m", "num_arch_ok_12_24m",
               "num_arch_rem_0_12m", "num_arch_written_off_0_12m", "num_arch_written_off_12_24m", "num_unpaid_bills",
               "recovery_debt", "sum_capital_paid_account_0_12m", "sum_capital_paid_account_12_24m", "sum_paid_inv_0_12m",
               "time_hours"]

cat_columns = ["account_status", "account_worst_status_0_3m", "account_worst_status_12_24m",
               "account_worst_status_3_6m", "account_worst_status_6_12m", "merchant_category", "merchant_group",
               "name_in_email", "status_last_archived_0_24m", "status_2nd_last_archived_0_24m",
               "status_3rd_last_archived_0_24m", "status_max_archived_0_6_months", "status_max_archived_0_12_months",
               "status_max_archived_0_24_months", "worst_status_active_inv"]

bool_columns = ["has_paid"]

# analyzing missing values
missing_values = df.isnull().sum()
percent = missing_values * 100 / len(df)

# visualizing missing values
msno.bar(df, figsize=(16, 6), sort="ascending", fontsize=12, color=(0, 0.40, 0.40))
plt.show()

# pie chart for percentage of missing values
labels = missing_values.index
sizes = percent.values
plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.axis('equal')
plt.title('Percentage of Missing Values')
plt.show()

# creating missing value dataframe
missvalue_df = pd.DataFrame({"Variable": labels, "Missing value": missing_values, "Missing value (%)": percent})
missvalue_df = missvalue_df.sort_values("Missing value (%)", ascending=False).reset_index(drop=True)
print(missvalue_df)

# calculate percentage of missing values
percent = df.isnull().sum() * 100 / len(df)
missvalue_df = pd.DataFrame({"Variable": df.columns, "Missing value": df.isnull().sum(), "Missing value (%)": percent})
missvalue_df = missvalue_df.sort_values("Missing value (%)", ascending=False).reset_index(drop=True)
print(missvalue_df)

#model fit value error
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df.reset_index()

x = df.drop(['uuid', 'default'], axis=1)
y = df['default']

# train and tests
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# numeric and categorical features
numeric_features = x.select_dtypes(include=['float64', 'int64']).columns
numeric_transformer = PowerTransformer(method='yeo-johnson')

categorical_features = x.select_dtypes(include=['object', 'bool']).columns
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

#model pipeline
model = make_pipeline(preprocessor, RandomForestClassifier())

#fit the model
model.fit(X_train, y_train)

#train and test predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#model evaluation
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

#training model 'model.joblib'
joblib.dump(model, 'model.joblib')

#loading model and identify
model = joblib.load('model.joblib')

#predictions of all data
predictions = model.predict_proba(x)

default_probabilities = predictions[:, 1]

# uuid ve pd DataFrame oluşturma
result_df = pd.DataFrame({'uuid': df['uuid'], 'pd': default_probabilities})

# 'result.csv' export
result_df.to_csv('result.csv', index=False)
