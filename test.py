#产生网格坐标
import numpy as np
import matplotlib.pyplot as plt
import create_data as cd

#产生[[600,600],[600,600]]的点，而且xx,yy 不相同
xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
print(xx.shape)
print(xx,yy)
print(xx.ravel().shape)
#把xx拉成(36000,)
print(xx.ravel())

#组成（36000,2）
grid = np.c_[xx.ravel(), yy.ravel()]
print(grid.shape)
print(grid)

X, Y_, Y_c = cd.generateds()
plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
#绘制等高线，predict,为高度   levels为划线的高度
#plt.contour(xx, yy, predict, levels=[0.5])
plt.show()