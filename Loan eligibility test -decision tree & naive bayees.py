#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
dataset = pd.read_csv("C:/Users/Ashwa/OneDrive/Desktop/Data project -SQL & BI/Loan eligibility ML/loan_data.csv")

# Strip and convert column names to lowercase
dataset.columns = dataset.columns.str.strip().str.lower()

# Handle missing values
dataset['gender'].fillna(dataset['gender'].mode()[0], inplace=True)
dataset['married'].fillna(dataset['married'].mode()[0], inplace=True)
dataset['dependents'].fillna(dataset['dependents'].mode()[0], inplace=True)
dataset['self_employed'].fillna(dataset['self_employed'].mode()[0], inplace=True)
dataset['loan_amount_term'].fillna(dataset['loan_amount_term'].mode()[0], inplace=True)
dataset['credit_history'].fillna(dataset['credit_history'].mode()[0], inplace=True)
dataset['loanamount'] = dataset['loanamount'].fillna(dataset['loanamount'].mean())

# Create new features
dataset['totalincome'] = dataset['applicantincome'] + dataset['coapplicantincome']
dataset['totalincome_log'] = np.log(dataset['totalincome'])
dataset['loanamount_log'] = np.log(dataset['loanamount'])

# Selecting features and target variable
x = dataset.iloc[:, np.r_[1:5, 9:11, 13, 15]].values
y = dataset.iloc[:, 12].values

# Splitting the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Convert x_train and x_test to DataFrame for easier handling
x_train = pd.DataFrame(x_train, columns=dataset.columns[np.r_[1:5, 9:11, 13, 15]])
x_test = pd.DataFrame(x_test, columns=dataset.columns[np.r_[1:5, 9:11, 13, 15]])

# Apply Label Encoding to the categorical features in training and test sets
labelencoder = LabelEncoder()

# Convert columns to string type before encoding
for col in x_train.columns[x_train.dtypes == 'object']:
    x_train[col] = x_train[col].astype(str)
    x_test[col] = x_test[col].astype(str)
    combined_data = pd.concat([x_train[col], x_test[col]], axis=0)
    labelencoder.fit(combined_data)
    x_train[col] = labelencoder.transform(x_train[col])
    x_test[col] = labelencoder.transform(x_test[col])

# Convert y_train and y_test to numeric
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)
y_test = labelencoder_y.transform(y_test)

# Feature Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Training the Decision Tree Classifier
DTC = DecisionTreeClassifier(criterion="entropy", random_state=0)
DTC.fit(x_train, y_train)

# Predicting the Test set results
y_pred = DTC.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)



# In[65]:


#naive_bayees
from sklearn.naive_bayes import GaussianNB
NBclassifier = GaussianNB()
NBclassifier.fit(x_train,y_train)


# In[66]:


y_pred = NBclassifier.predict(x_test)
y_pred


# In[16]:


print("The accuracy of naive bayes algorithm is ",accuracy_score(y_pred,y_test))


# In[ ]:





# In[24]:


import pandas as pd


testdata = pd.read_excel("C:/Users/Ashwa/Downloads/archive (1)/test data.xlsx")


# In[25]:


testdata.head()


# In[26]:


testdata.info()


# In[27]:


testdata.isnull().sum()


# In[31]:


testdata["Gender"].fillna(testdata["Gender"].mode()[0], inplace=True)
testdata["Married"].fillna(testdata["Married"].mode()[0], inplace=True)
testdata["Dependents"].fillna(testdata["Dependents"].mode()[0], inplace=True)
testdata["Self_Employed"].fillna(testdata["Self_Employed"].mode()[0], inplace=True)
testdata["Loan_Amount_Term"].fillna(testdata["Loan_Amount_Term"].mode()[0], inplace=True)
testdata["Credit_History"].fillna(testdata["Credit_History"].mode()[0], inplace=True)


# In[32]:


testdata.isnull().sum()


# In[39]:


import pandas as pd
import matplotlib.pyplot as plt

testdata.boxplot(column="LoanAmount")

plt.show()


# In[42]:


testdata.boxplot(column="ApplicantIncome")

plt.show()


# In[44]:


testdata.LoanAmount = testdata["LoanAmount"].fillna(testdata.LoanAmount.mean())


# In[45]:


testdata["LoanAmount_log"]=np.log(testdata["LoanAmount"])


# In[46]:


testdata.isnull().sum()


# In[47]:


testdata["TotalIncome"] = testdata["ApplicantIncome"] + testdata["CoapplicantIncome"]
testdata["TotalIncome_log"] = np.log(testdata["TotalIncome"])


# In[48]:


testdata.isnull().sum()


# In[49]:


testdata.head()


# In[50]:


test = testdata.iloc[:,np.r_[1:5,9:11,13:15]].values


# In[57]:


labelencoder_X = LabelEncoder()


# In[58]:


for i in range(0,5):
    test[:, i] = labelencoder_X.fit_transform(test[:, i].astype(str))


# In[59]:


test[:, 7] = labelencoder_X.fit_transform(test[:, 7].astype(str))


# In[60]:


test


# In[69]:


print(type(test))  # Check the type of test
print(test)        # Print the contents to inspect


# In[70]:


if isinstance(test, pd.DataFrame):
    test = test.to_numpy()


# In[72]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Convert DataFrame to NumPy array if needed
if isinstance(test, pd.DataFrame):
    test = test.to_numpy()

# Convert all values to numeric, forcing non-numeric values to NaN
test = pd.DataFrame(test).apply(pd.to_numeric, errors='coerce').to_numpy()

# Replace NaN values with 0 or another appropriate value
test = np.nan_to_num(test, nan=0)

# Initialize and apply the scaler
ss = StandardScaler()
test = ss.fit_transform(test)


# In[73]:


test


# In[78]:


ss = StandardScaler()
test = ss.fit_transform(test)


# In[81]:


NBclassifier.fit(x_test,y_test)


# In[82]:


pred = NBclassifier.predict(test)


# In[83]:


pred


# In[ ]:


#1 implies the customer is eligible for the loan , 0 implies that he/she is not eligible

