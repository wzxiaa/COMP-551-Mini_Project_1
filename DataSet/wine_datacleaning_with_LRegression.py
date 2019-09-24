import math
import numpy as np
class LogisticRegression(object):
    
    def __init__(self):
        pass
    
    def logictic_function (self, a):
        try:
            expa = math.exp(-a)
        except OverflowError:
            expa = float('inf')
        sigma = 1 / (1+expa)
        return sigma
    
    def fit (self, x, y, rate, iteration):
        i = 1
        w = np.zeros((len(x[0]),1))
        w_next = np.zeros((len(x[0]),1))
        
        
        while i<iteration: 
            w = w_next
            
            derivative = np.zeros((len(x[0]),1))
            for j in range (0, len(x)):
                xj = np.array([[element] for element in x[j]])
                sigma = self.logictic_function ( np.matmul(np.transpose(w), xj) )
                derivative = np.add (derivative,  np.array( [[element] for element in xj @ (y[j] - sigma)])  )
                         
            w_next = np.add(w, rate(i) * derivative)
            i = i + 1
            
        self.w = w_next
    
    
    def predict (self, x):
        return np.array(list(map(lambda xi: [1] if np.sum([xi] @ self.w) > 0 else [0], x)))


# In[13]:


# LRModel = LogisticRegression()
# LRModel.fit (x, y, lambda n: 0.001/(n+1), 3000)


# # In[15]:


# LRModel.predict(x)


# # In[ ]:





# # In[16]:


# prediction = LRModel.predict(x)
# counter = 0
# for i in range(0, len(prediction)):
#     if prediction[i]==y[i]:
#         counter += 1
# print (counter)


# # In[ ]:




