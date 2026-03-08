import numpy as np
import matplotlib.pyplot as plt

# Parameters
r = 3.6
x0 = 0.2  # initial population
n = 50    # number of generations

# Generate logistic map time series
x = np.zeros(n)
x[0] = x0
for i in range(1, n):
    x[i] = r * x[i-1] * (1 - x[i-1])

# Plot styling
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(10, 5))

# Plot time series with lines and markers
ax.plot(x, 'o-', color='darkorange', markersize=6, linewidth=2, 
        markeredgecolor='black', markeredgewidth=1, label=f'r = {r}')

# Highlight the period-4 cycle after transients
ax.axvspan(30, 50, alpha=0.15, color='gray', label='Steady cycle region')

# Axis labels and title
ax.set_xlabel('Generation (n)', fontsize=13)
ax.set_ylabel('Population xₙ', fontsize=13)
ax.set_title(f'Logistic Map: Period-4 Cycle at r = {r}', fontsize=15, weight='bold')

# Grid and ticks
ax.grid(True, alpha=0.3)
ax.set_xticks(np.arange(0, 51, 5))
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_ylim(0, 1)

# Legend
ax.legend(fontsize=12, loc='upper right')

plt.tight_layout()
plt.show()