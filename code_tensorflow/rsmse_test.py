import datetime as dt
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

np.random.seed(0)
var1 = np.random.randn(10000)*1.1
var1_smse = np.square(np.where(var1 > 0, var1, 0))

np.random.seed(1)
var2 = np.random.randn(20000)
var2_smse = np.square(np.where(var2 > 0, var2, 0))

t_test = ttest_ind(var2, var1, equal_var=False, alternative="less")
t_test_smse = ttest_ind(var2_smse, var1_smse, equal_var=False, alternative="less")

var1_smse_mean = np.mean(var1_smse)
var2_smse_mean = np.mean(var2_smse)

var1_smse_stde = np.std(var1_smse, ddof=1)/np.sqrt(10000)
var2_smse_stde = np.std(var2_smse, ddof=1)/np.sqrt(20000)

t = (var2_smse_mean-var1_smse_mean)/(np.sqrt(np.square(var2_smse_stde)+np.square(var1_smse_stde)))

var1_rsmse = np.sqrt(var1_smse_mean)
var2_rsmse = np.sqrt(var2_smse_mean)

var1_rsmse_stde = np.std(var1_smse, ddof=1)/np.sqrt(10000)
var2_rsmse_stde = np.std(var2_smse, ddof=1)/np.sqrt(20000)

t_r = (var2_rsmse-var1_rsmse)/(np.sqrt(np.square(var2_smse_stde)+np.square(var1_smse_stde)))

print(t_test)
print(t)
print(t_r)
print(t_test_smse)

print("|-----------------------------------------------------------------------Comparison of Mean Hedging Error------------------------------------------------------------------------|")
print("|\t\tTransformer\t\t|\t\tGRU\t\t|\t\tLSTM\t\t|\t\tFFNN\t\t|\t\tDelta Hedge\t\t|")
print("|---------------------------------------|-------------------------------|-------------------------------|-------------------------------|---------------------------------------|")
print("|\t{:.4f} +- {:.4f}\t\t|\t{:.4f} +- {:.4f}\t|\t{:.4f} +- {:.4f}\t|\t{:.4f} +- {:.4f}\t|\t{:.4f} +- {:.4f}\t\t|".format(np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn()))
print("|---------------------------------------|-------------------------------|-------------------------------|-------------------------------|---------------------------------------|")