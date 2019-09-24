import numpy as np

class LDAClassifier(object):
    
    def __init__(self, psudo_inv=False):
        """
        Provide an option of psudo inverse just in case not invertable
        """
        self.psudo_inv = psudo_inv 

    def fit(self, x, y):
        """
        Input dim of x is number of case x number of features
                     y is number of case x 1
        Output       None
        """
        self.n = np.array(y).size
        self.n1 = np.sum(y)
        self.n0 = self.n - self.n1
        self.p0 = self.n0 / self.n
        self.p1 = self.n1 / self.n
        # This is, however, transpose of u0 and u1, since its shape is (m,)
        self.u0 = np.sum([xi for xi, yi 
                             in list(zip(x, y)) 
                             if yi == 0], axis=0) / self.n0
        self.u1 = np.sum([xi for xi, yi 
                             in list(zip(x, y)) 
                             if yi == 1], axis=0) / self.n1
        # since x supposed to be a column vector of rv x1 to xm
        # this (x - ui)(x - ui)T is actually computed with transpose of
        # x and ui (x passed as ncase x nfeature (n x m))
        # so the computation becomes (xT - uiT)T(xT - uiT) 
        xxT = lambda v: np.transpose(v) @ np.array(v)
        # test: 
        # print(np.sum([xxT([x[0] - self.u0]), \
        #               xxT([x[3] - self.u0])], axis=0))
        self.sigma = (np.sum([xxT(np.array([xi - self.u0]))
                              for xi, yi 
                              in list(zip(x, y))
                              if yi == 0], axis=0) + 
                      np.sum([xxT(np.array([xi - self.u1]))
                              for xi, yi 
                              in list(zip(x, y))
                              if yi == 1], axis=0)) / (self.n - 2)
        self.sigma_inv = np.linalg.pinv(self.sigma)                          if self.psudo_inv                          else np.linalg.inv(self.sigma)
        # test: 
        # print(np.sum(np.array([self.u1]) @ 
        #              self.sigma_inv @ 
        #              np.transpose([self.u1])))
        self.w0 = np.log(self.p1 / self.p0) -                   (1 / 2) * np.sum(np.array([self.u1]) @                                    self.sigma_inv @                                    np.transpose([self.u1])) +                   (1 / 2) * np.sum(np.array([self.u0]) @                                    self.sigma_inv @                                    np.transpose([self.u0]))
        # test: 
        # print(self.sigma_inv @ np.transpose([self.u1 - self.u0]))
        self.w = self.sigma_inv @                  np.transpose([self.u1 - self.u0])

    def predict(self, x):
        return np.array(list(map(lambda xi: [1] 
                                 if self.w0 + np.sum([xi] @ self.w) > 0 
                                 else [0], x)))



# In[ ]:




