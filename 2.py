import matplotlib.pyplot as plt
import numpy as np


# 发光二极管

y_1 = range(1970,1994)
n_1 = [7,13,21,26,31,28,39,38,39,43,74,76,80,112,122,119,145,116,119,110,117,128,109,120]
c_1 = [0,0,1,1,0,3,1,3,4,3,4,9,6,2,4,3,7,6,4,8,14,35,42,12]
# if len(y)!=len(n): print("errors")
# for i in range(len(y)):
#    print(y[i],n[i])
plt.subplot(3,2,1)
plt.plot(y_1,n_1)
plt.subplot(3,2,2)
plt.plot(y_1,c_1)

# 基因编辑
y_2 = range(1983,2013)
n_2 = [1082,1462,1747,1937,2247,2576,2781,3102,3587,3999,4457,6091,7295,8341,9917,11507,14591,19022,23160,26945,32254,38904,44480,53032,54394,54943,57772,58498,59655,61484]
c_2 = [248,173,548,275,617,512,273,439,422,569,681,658,374,2326,546,744,722,1115,1275,789,1749,1110,2642,2178,7514,693,624,789,651,1088]
print(len(y_2),len(n_2))

plt.subplot(3,2,3)
plt.plot(y_2,n_2)
plt.subplot(3,2,4)
plt.plot(y_2,c_2)

# 超导
y_3 = range(1986,2023)
n_3 = [146,386,568,547,510,570,533,470,511,442,437,384,377,467,526,611,619,733,701,762,890,938,962,1013,902,968,975,989,1016,1010,1040,1100,1111,1081,1032,1136,1039]

plt.subplot(3,2,5)
plt.plot(y_3,n_3)
plt.show()