import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


uniform_data = np.random.rand(20, 12000)
ax = sns.heatmap(uniform_data)
plt.show()
