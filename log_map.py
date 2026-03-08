import numpy as np
import matplotlib.pyplot as plt

# Parameters
r = 3.5
x = np.linspace(0, 1, 500)
y = r * x * (1 - x)

# Set up the plot with a clean style
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the logistic curve and diagonal line
ax.plot(x, y, label=r'$f(x) = r x (1 - x)$', color='crimson', linewidth=2.5)
ax.plot(x, x, label=r'$y = x$', linestyle='--', color='navy', linewidth=2)

# Add the 2D plane axes (thick axes lines at x=0 and y=0)
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)

# Set limits and ticks
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks(np.arange(0, 1.1, 0.1))
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.tick_params(direction='in', length=6, width=1.5, labelsize=10)

# Labels and title with LaTeX for style
ax.set_xlabel(r'$x$', fontsize=14)
ax.set_ylabel(r'$f(x)$', fontsize=14)
ax.set_title(f'Logistic Map Function with $r = {r}$', fontsize=16, weight='bold')

# Legend with frame
ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, loc='upper left')

# Tight layout for clean spacing
plt.tight_layout()
plt.show()