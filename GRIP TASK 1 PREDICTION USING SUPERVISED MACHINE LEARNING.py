#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION
# 

# # DATA SCIENCE AND BUSINESS ANALYTICS INTERNSHIP
# 

# # TASK 1-PREDICTION USING SUPERVISED MACHINE LEARNING

# Date-14th March'2022
# By-Shreya Ghosh

# OBJECTIVE OF THE TASK - Predicting the number of a student based on his/her number of study hours
# 
# Linear Regression with Supervised Machine learning will be used to complete the task
# 
# The task will be processed through the following steps:-

# # STEPS:
STEP 1-IMPORTING THE DATASET FROM EXCEL
STEP 2-PREPARING THE DATASET
STEP 3-TRAINING THE MODEL
STEP 4-TESTING THE MODEL/MAKING PREDICTIONS
STEP 5-VISUALIZATION OF THE MODEL
STEP 6-EVALUATING THE MODEL/DIAGONISTIS
# # STEP 1 IMPORTING THE DATASET

# In[1]:


#Lets first import all the libraries that will be required throughout the process
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# In[2]:


#Importing the dataset
stdy=pd.read_csv("C:/Users/Shreya/OneDrive/Desktop/GRIP FILE/student_scores - student_scores.csv")
stdy


# In[3]:


#Now let us observe the properties of the dataset

stdy.shape


# In[4]:


stdy.columns


# In[5]:


#To check for any missing or null values in the dataset

stdy.isna().sum()

# We can observe that the dataset donot have any missing values 


# # STEP 2 PREPARING THE DATASET

# We will first divide the data set into independent (Study hours or the X variable) and dependent (Marks or the Y variable) variable.

# In[6]:


X=stdy.iloc[:,:-1].values
X


# In[7]:


Y=stdy.iloc[:,1:].values
Y


# Now that we have defined our X and Y variable we will split the data set into testing and training data

# In[8]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)


# In[9]:


X_train.shape,X_test.shape,Y_train.shape,Y_test.shape


# # STEP 3 TRAINING THE DATASET¶
In this step we will be training our X_train and Y_train dataset
# In[10]:


model= LinearRegression()
model


# In[11]:


training=model.fit(X_train,Y_train)
print("The training of the model is complete",training)


# In[12]:


training.coef_


# In[13]:


training.intercept_


# # STEP 4 TESTING THE MODEL/PREDICTINGTHE MODEL

# In this step we will validate the model through prediction

# In[14]:


pred=model.predict(X_test)
pred


# In[34]:


Actual=Y_test
Actual


# In[39]:


x=8
y=2.89354722+9.54488104*x
print("The predicted score of the student is",y)


# # STEP 5 VISUALIZATION OF THE MODEL

# In[22]:


line=training.coef_*X+training.intercept_


# In[23]:


plt.rcParams["figure.figsize"]=[20,9]
plt.xlabel("Study hours")
plt.ylabel("Exam Score")
plt.scatter(X_train,Y_train,color="red")
plt.plot(X,line,color="blue")
plt.grid()
plt.show()


# In[24]:


#Plotting the test dataset

plt.rcParams["figure.figsize"]=[20,12]
plt.scatter(X_test,Y_test,color="red")
plt.plot(X,line,color="blue")
plt.xlabel("Study hours(test data)")
plt.ylabel("Exam Score(test data)")
plt.grid()
plt.show()


# # STEP 6 EVALUATING THE MODEL/DIAGONISTIS

# In this step we will evaluate the accuracy of the model

# In[25]:


# We will use R2 score and Errors to measure the accuracy of the model

r2_score(Y_test,pred)


# In[26]:


ERRORS=1-r2_score(Y_test,pred)
ERRORS
# The model have pretty low error percentage

