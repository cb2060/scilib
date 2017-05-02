import matplotlib.pyplot as plt
import numpy as np
plt.plot(np.random.randn(30), linestyle='--', color='g')
plt.savefig('myfirstpic.png', dpi=400)
