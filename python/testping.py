# -*- coding: utf-8 -*-
"""
Created on Thu Apr 02 20:52:06 2015

@author: Bernat
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#%%
lF = 0.5
lR = 1800  

def PredisDown(T,lRap, alfa):
  return ((1-np.e**(-T*lF))/(1-np.e**(-T*lF)*(1-np.e**(-T*lRap))) > alfa)

class Worker(object):
  def __init__(self,l1,l2):
    self.lF = l1
    self.lR = l2
    self.t = np.random.exponential(1./self.lR)
    self.failed = False
  def Start(self):
    self.t = np.random.exponential(1./self.lR)
  def Check(self,T):
    self.failed = self.failed or np.random.binomial(1,1-np.e**(-T*lF))
    return (self.failed or T<self.t)

class Master(object):
  def __init__(self,l1=lF,l2=lR, n=100,alpha=0.95, lRmax = None):
    self.lF = l1
    self.lR = np.array([l2/10. for i in range(n)])
    self.workers = [Worker(l1,l2) for i in range(n)]
    self.predfail = [False for i in range(n)]
    self.timesfail = np.array([0 for i in range(n)])
    self.alpha = alpha
    self.lRmax = lRmax
    if lRmax is None:
      self.lRmax = 3*l2
  def Check(self,T):
    idx = np.where([not b for b in self.predfail])[0]
    for i in idx:
      if self.workers[i].Check(T[i]):
        self.timesfail[i]+=1
        self.predfail[i] = self.PredictDown(T[i],self.lR[i])
      else:
        self.timesfail[i]=0
        self.Update(T[i],i)
        
  def Update(self,T,i):
    self.lR[i] = min((self.lR[i]+1./T)/2.,self.lRmax)
    
  def PredictDown(self,T,lam):
    return ((1-np.e**(-T*self.lF))/(1-np.e**(-T*lF)*(1-np.e**(-T*lam)))>self.alpha)
  def Test(self):
    T = (0.01/self.lR)*2**(self.timesfail)
    idx = np.where([not b for b in self.predfail])[0]
    idx2 = np.where([b == 0 for b in self.timesfail])[0]
    idx3 = [a and b for a,b in zip(idx,idx2)]
    for i in idx3:
      self.workers[i].Start()
    self.Check(T)

class StupidMaster(Master):
  def PredictDown(self,T,lam):
    return (1-np.e**(-T*self.lF) > np.e**(-T*lam))
    
#%%
master1 = Master()
master2 = StupidMaster()
for i in range(100):
  master1.Test()
  master2.Test()
realfails1 = [worker.failed for worker in master1.workers]
realfails2 = [worker.failed for worker in master2.workers]
FP1 = float(sum([f and rf for f,rf in zip(master1.predfail,realfails1)]))/sum(master1.predfail)
FP2 = float(sum([f and rf for f,rf in zip(master2.predfail,realfails2)]))/sum(master2.predfail)

#%%
FP = []
maxwait = []
alphas = np.linspace(0.95,1.0,100)
#%%
for alpha in alphas:  
  master = Master(alpha = alpha)
  for i in range(1000):
    master.Test()
  realfails = [not worker.failed for worker in master.workers]
  try:
    FP.append(float(sum([f and rf for f,rf in zip(master.predfail,realfails)]))/sum(master.predfail))
  except ZeroDivisionError:
    FP.append(0.)
  maxwait.append(max(master.timesfail))

#%%
fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(alphas,FP)
plt.title("False Positives probability")
ax = fig.add_subplot(212)
ax.plot(alphas[:-1],maxwait[:-1])
plt.title("Number of pings")