#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np


# In[6]:


def accuracy(y, predictedY):
    count =0;
    arrayLen = len(y);
    for i in range(0,arrayLen):
        #print(y[i]==predictedY[i]);
        if(y[i]==predictedY[i]):
            count=count +1;
    #print(count)
    error = arrayLen-count
    accuracy = count / arrayLen;
    return accuracy;


# In[23]:


def f1Score(y, predictedY,positiveOutcome, negativeOutcome):
    arrayLen = len(y);
    countPositive = predictedY.count(positiveOutcome); 
    countTruePositive = 0;
    countFalseNegative =0;

    for i in range(0,arrayLen):
        if(y[i]==positiveOutcome and predictedY[i] ==positiveOutcome):
            countTruePositive= countTruePositive+1;
    
    for i in range(0,arrayLen):
        if(y[i]!=predictedY[i] and predictedY[i]==negativeOutcome):
            countFalseNegative = countFalseNegative+1;
    
    #calculate recall(# TP/TP+FN)
    recall = countTruePositive/(countTruePositive+countFalseNegative);
          
    #calculate precision (TP/(TP+FP))
    precision = countTruePositive/countPositive; 
    
    #calculate f1 score
    f1Score = (2*precision*recall)/(precision+recall)
    return f1Score;
    


# In[24]:


# y=[1,0,0,1,1,1] #real 
# y1=[1,1,0,1,0,0] #predicted

# print(f1Score(y,y1,1,0));


# In[ ]:





# In[ ]:




