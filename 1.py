import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
from scipy.interpolate import interp1d


def func1(x,a,b,c):
    return a*pow(math.e,b*x)+c

def func2(x,a,b,c):
    return a*np.arctan(b*x)+c

def func3(x,a,b,c):
    return a*x/(np.abs(x)+b)+c

def func4(x,a,b,c):
    return 1/(a*pow(b,x)+c) 

x = []
y = [0.919,0.905,0.889,0.871,0.850,0.826,0.800,0.771,0.739,0.704,0.667,0.627,0.586,0.543,0.500,0.457,0.414,0.373,0.333,0.296,0.261,0.229,0.200,0.174,0.150,0.129,0.111,0.095,0.081,0.069]
for i in range(-15,15):
    x.append(i*0.25)

'''
plt.title('Problem 1')
plt.scatter(x,y)
plt.show()
'''

# 多项式拟合
# 二次多项式 R=0.03352  三次多项式 R=0.00259  四次：0.00238  五次：0.00169
'''
if len(x)!=len(y): print('Length error')
else:
    p=np.polyfit(x,y,2)
    xn = np.linspace(-4,4,1000)
    yn = np.poly1d(p)
    plt.plot(xn,yn(xn),x,y,'o')
    plt.show()
    print(yn)
    yfit = yn(x)
    yr = y-yfit
    R = sum(pow(yr,2))
    print("R=",R)
'''
'''
# 插值
func = interp1d(x,y,kind='cubic')
x_new = np.linspace(start=min(x),stop=max(x),num=len(x))
y_new = func(x_new)
plot1=plt.plot(x,y,'r')
plt.show()
'''

# R= 0.03806972732060332
'''
popt,pcov = curve_fit(func1,x,y,method='lm',maxfev=8866)
plot1=plt.plot(np.array(x),y,'r*')
plot2=plt.plot(np.array(x),[func1(i,*popt) for i in x],'b')
yr = np.array([func1(i,*popt) for i in x])-y
R=sum(pow(yr,2))
print("R=",R)
plt.show()
'''
'''
popt,pcov = curve_fit(func2,x,y,method='lm',maxfev=8866)
plot1=plt.plot(np.array(x),y,'r*')
plot2=plt.plot(np.array(x),[func2(i,*popt) for i in x],'b')
yr = np.array([func2(i,*popt) for i in x])-y
R=sum(pow(yr,2))
print("R=",R)
plt.show()
'''
'''
popt,pcov = curve_fit(func3,x,y,method='lm',maxfev=8866)
plot1=plt.plot(np.array(x),y,'r*')
plot2=plt.plot(np.array(x),[func3(i,*popt) for i in x],'b')
yr = np.array([func3(i,*popt) for i in x])-y
R=sum(pow(yr,2))
print("R=",R)
print(*popt)
plt.show()
'''

'''
p=np.polyfit(x,y,2)
xn = np.linspace(-4,4,1000)
yn = np.poly1d(p)
plt.plot(xn,yn(xn),x,y,'o')
plt.show()
print(yn)
yfit = yn(x)
yr = y-yfit
R = sum(pow(yr,2))
print("R=",R)
'''
'''
popt,pcov = curve_fit(func3,x,y,method='lm',maxfev=8866)
plot1=plt.plot(np.array(x),y,'r*')
plot2=plt.plot(np.array(x),[func3(i,*popt) for i in x],'b')
yr = np.array([func3(i,*popt) for i in x])-y
R=sum(pow(yr,2))
print("R=",R)
print(*popt)
plt.show()
'''
popt,pcov = curve_fit(func3,x,y,method='lm',maxfev=8866)
plot1=plt.plot(np.array(x),y,'r*')
plot2=plt.plot(np.array(x),[func3(i,*popt) for i in x],'b')
yr = np.array([func3(i,*popt) for i in x])-y
R=sum(pow(yr,2))
print("R=",R)
print(*popt)
plt.show()