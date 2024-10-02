#Chebyquad Visualizer
import matplotlib.pyplot as plt
import numpy as np
#exact line search

plt.figure(1)
x1 = [1, 2, 3, 4]
y1 = np.sort(optimizer_GB4.xhist[-1])
y2 = np.sort(optimizer_BB4.xhist[-1])
y3 = np.sort(optimizer_SB4.xhist[-1])
y4 = np.sort(optimizer_DFP4.xhist[-1])
y5 = np.sort(optimizer_BFGS4.xhist[-1])
y6= np.sort(xmin4)
plt.title('Chbeyquad n=4') # 折线图标题
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示汉字
plt.xlabel('size of input number') # x轴标题
plt.ylabel("Output") # y轴标题
plt.plot(x1, y1, marker='o', markersize=3) # 绘制折线图，添加数据点，设置点的大小
plt.plot(x1, y2, marker='o', markersize=3)
plt.plot(x1, y3, marker='o', markersize=3)
plt.plot(x1, y4, marker='o', markersize=3)
plt.plot(x1, y5, marker='o', markersize=3)
plt.plot(x1, y6, marker='x', markersize=3)
plt.legend(['Good Broyden','Bad Broyden','Symmetric Broyden','DFP','BFGS','fmin_bfgs'])


plt.figure(2)
x2 = [1, 2, 3, 4,5,6,7,8]
y1 = np.sort(optimizer_GB8.xhist[-1])
y2 = np.sort(optimizer_BB8.xhist[-1])
y3 = np.sort(optimizer_SB8.xhist[-1])
y4 = np.sort(optimizer_DFP8.xhist[-1])
y5 = np.sort(optimizer_BFGS8.xhist[-1])
y6= np.sort(xmin8)
plt.title('Chbeyquad n=8') # 折线图标题
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示汉字
plt.xlabel('size of input number') # x轴标题
plt.ylabel("Output") # y轴标题
plt.plot(x2, y1, marker='o', markersize=3) # 绘制折线图，添加数据点，设置点的大小
plt.plot(x2, y2, marker='o', markersize=3)
plt.plot(x2, y3, marker='o', markersize=3)
plt.plot(x2, y4, marker='o', markersize=3)
plt.plot(x2, y5, marker='o', markersize=3)
plt.plot(x2, y6, marker='x', markersize=3)
plt.legend(['Good Broyden','Bad Broyden','Symmetric Broyden','DFP','BFGS','fmin_bfgs'])


plt.figure(3)
x3 = [1, 2, 3, 4,5,6,7,8,9,10,11]
y1 = np.sort(optimizer_GB11.xhist[-1])
y2 = np.sort(optimizer_BB11.xhist[-1])
y3 = np.sort(optimizer_SB11.xhist[-1])
y4 = np.sort(optimizer_DFP11.xhist[-1])
y5 = np.sort(optimizer_BFGS11.xhist[-1])
y6= np.sort(xmin11)
plt.title('Chbeyquad n=11') # 折线图标题
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示汉字
plt.xlabel('size of input number') # x轴标题
plt.ylabel("Output") # y轴标题
plt.plot(x3, y1, marker='o', markersize=3) # 绘制折线图，添加数据点，设置点的大小
plt.plot(x3, y2, marker='o', markersize=3)
plt.plot(x3, y3, marker='o', markersize=3)
plt.plot(x3, y4, marker='o', markersize=3)
plt.plot(x3, y5, marker='o', markersize=3)
plt.plot(x3, y6, marker='x', markersize=3)
plt.legend(['Good Broyden','Bad Broyden','Symmetric Broyden','DFP','BFGS','fmin_bfgs'])
