import matplotlib.pyplot as plt
import numpy as np


# Primo set di dati
x1 = np.array([1, 2, 3, 4, 5])
y1 = np.power(x1, 2)  # y = x**2
e1 = np.array([1.5, 2.6, 3.7, 4.6, 5.5])

# Secondo set di dati
x2 = np.array([1, 2, 3, 4, 5])
y2 = np.power(x2, 3)  # y = x**3
e2 = np.array([0.5, 1.6, 2.7, 3.6, 4.5])

# Creare il grafico
plt.errorbar(x1, y1, e1, linestyle='None', marker='^', label='Set 1')
plt.errorbar(x2, y2, e2, linestyle='None', marker='o', label='Set 2')

# Aggiungere legenda
plt.legend()

# Mostrare il grafico
plt.show()