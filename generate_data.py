import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置随机种子保证可重复性
np.random.seed(42)

# 生成50000个x值（0到10之间均匀分布）0
x = np.linspace(0, 10, 50000)

# 真实线性关系：y = 3x + 5 + 噪声
true_slope = 3.0
true_intercept = 5.0
noise = np.random.normal(0, 1.5, size=len(x))  # 均值为0，标准差1.5的正态分布噪声
y = true_slope * x + true_intercept + noise

# 创建DataFrame并保存为CSV
data = pd.DataFrame({'x': x, 'y': y})
data.to_csv('linear_data.csv', index=False)
print("数据已保存为 linear_data.csv")
