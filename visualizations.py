import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['figure.dpi'] = 100

# Logistic map function
def logistic_map(x, r):
    return r * x * (1 - x)

def plot_cobweb(r, x0, iterations=50, save=False):
    """Create a beautiful cobweb plot"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    x = np.linspace(0, 1, 1000)
    y = logistic_map(x, r)
    
    # Plot the map and diagonal
    ax.plot(x, y, 'b-', linewidth=2.5, label=f'f(x) = {r}x(1-x)')
    ax.plot(x, x, 'k--', linewidth=1.5, alpha=0.7, label='y = x')
    
    # Cobweb iterations
    xn = x0
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, iterations))
    
    for i in range(iterations):
        x_next = logistic_map(xn, r)
        color = colors[i]
        
        # Vertical line
        ax.plot([xn, xn], [xn, x_next], color=color, linewidth=1.5, alpha=0.8)
        # Horizontal line
        ax.plot([xn, x_next], [x_next, x_next], color=color, linewidth=1.5, alpha=0.8)
        
        xn = x_next
    
    # Mark the starting point
    ax.plot(x0, logistic_map(x0, r), 'ro', markersize=10, label='Start', zorder=5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('$x_n$', fontsize=16)
    ax.set_ylabel('$x_{n+1}$', fontsize=16)
    ax.set_title(f'Cobweb Plot for r = {r:.3f}', fontsize=18, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    if save:
        plt.savefig(f'cobweb_r_{r:.3f}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_time_series(r, x0, iterations=100, save=False):
    """Create an elegant time series plot"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x_vals = [x0]
    x = x0
    for _ in range(iterations):
        x = logistic_map(x, r)
        x_vals.append(x)
    
    n = range(iterations + 1)
    
    # Create color gradient based on value
    colors = plt.cm.plasma(np.array(x_vals))
    
    # Plot with gradient effect
    for i in range(len(n)-1):
        ax.plot(n[i:i+2], x_vals[i:i+2], color=colors[i], linewidth=2, alpha=0.8)
    
    # Add scatter points
    ax.scatter(n, x_vals, c=x_vals, cmap='plasma', s=30, zorder=5, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Iteration (n)', fontsize=16)
    ax.set_ylabel('$x_n$', fontsize=16)
    ax.set_title(f'Time Series Evolution for r = {r:.3f}', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='plasma'), ax=ax)
    cbar.set_label('$x_n$ value', fontsize=12)
    
    # Add statistics box
    stats_text = f'Mean: {np.mean(x_vals):.4f}\nStd: {np.std(x_vals):.4f}\nMin: {np.min(x_vals):.4f}\nMax: {np.max(x_vals):.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    if save:
        plt.savefig(f'timeseries_r_{r:.3f}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_bifurcation(r_min=2.5, r_max=4.0, r_steps=5000, iterations=2000, last=500, save=False):
    """Create a high-resolution bifurcation diagram"""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    r_values = np.linspace(r_min, r_max, r_steps)
    x = 0.1 * np.ones(r_steps)
    
    # Store points for density coloring
    all_r = []
    all_x = []
    
    # Burn-in period
    for i in range(iterations - last):
        x = logistic_map(x, r_values)
    
    # Collect points
    for i in range(last):
        x = logistic_map(x, r_values)
        all_r.extend(r_values)
        all_x.extend(x)
    
    # Create hexbin plot for better visualization of density
    hb = ax.hexbin(all_r, all_x, gridsize=500, cmap='inferno', mincnt=1, alpha=0.8)
    
    # Add colorbar
    cbar = fig.colorbar(hb, ax=ax, label='Point Density')
    
    # Highlight important regions
    ax.axvline(x=3.0, color='cyan', linestyle='--', alpha=0.5, label='Period doubling starts')
    ax.axvline(x=3.5699456, color='red', linestyle='--', alpha=0.5, label='Onset of chaos')
    ax.axvline(x=3.8284, color='green', linestyle='--', alpha=0.5, label='Period 3 window')
    
    ax.set_xlim(r_min, r_max)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Bifurcation parameter r', fontsize=16)
    ax.set_ylabel('x', fontsize=16)
    ax.set_title('Bifurcation Diagram of the Logistic Map', fontsize=20, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    if save:
        plt.savefig('bifurcation_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_lyapunov(r_min=2.5, r_max=4.0, r_steps=2000, iterations=1000, save=False):
    """Create a detailed Lyapunov exponent plot"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    r_values = np.linspace(r_min, r_max, r_steps)
    x = 0.5 * np.ones(r_steps)
    lyapunov = np.zeros(r_steps)
    
    # Calculate Lyapunov exponent
    for i in range(iterations):
        x = logistic_map(x, r_values)
        lyapunov += np.log(np.abs(r_values * (1 - 2 * x)))
    
    lyapunov /= iterations
    
    # Main Lyapunov plot
    ax1.plot(r_values, lyapunov, 'b-', linewidth=1.5)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='λ = 0')
    
    # Color regions based on stability
    ax1.fill_between(r_values, 0, lyapunov, where=(lyapunov > 0), color='red', alpha=0.3, label='Chaotic (λ > 0)')
    ax1.fill_between(r_values, lyapunov, 0, where=(lyapunov < 0), color='blue', alpha=0.3, label='Periodic (λ < 0)')
    
    ax1.set_xlim(r_min, r_max)
    ax1.set_ylim(-2, 1)
    ax1.set_ylabel('Lyapunov Exponent λ', fontsize=14)
    ax1.set_title('Lyapunov Exponent Spectrum', fontsize=18, fontweight='bold')
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Bifurcation diagram reference in bottom subplot
    r_bif = np.linspace(r_min, r_max, 2000)
    x_bif = 0.1 * np.ones(2000)
    
    # Burn-in
    for i in range(500):
        x_bif = logistic_map(x_bif, r_bif)
    
    # Plot bifurcation reference
    for i in range(100):
        x_bif = logistic_map(x_bif, r_bif)
        ax2.scatter(r_bif, x_bif, c='black', s=0.1, alpha=0.1)
    
    ax2.set_xlim(r_min, r_max)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Bifurcation parameter r', fontsize=14)
    ax2.set_ylabel('x', fontsize=14)
    ax2.set_title('Reference Bifurcation Diagram', fontsize=14)
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    if save:
        plt.savefig('lyapunov_spectrum.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_comprehensive_analysis(r_values=[2.8, 3.2, 3.5, 3.83], x0=0.2, iterations=100, save=False):
    """Create a comprehensive multi-panel analysis"""
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Main bifurcation diagram
    ax_bif = fig.add_subplot(gs[:, 0])
    
    # Generate bifurcation data
    r_bif = np.linspace(2.5, 4.0, 4000)
    x_bif = 0.1 * np.ones(4000)
    for i in range(1000):
        x_bif = logistic_map(x_bif, r_bif)
    
    all_r, all_x = [], []
    for i in range(200):
        x_bif = logistic_map(x_bif, r_bif)
        all_r.extend(r_bif)
        all_x.extend(x_bif)
    
    ax_bif.hexbin(all_r, all_x, gridsize=400, cmap='viridis', mincnt=1, alpha=0.7)
    ax_bif.set_xlabel('r', fontsize=14)
    ax_bif.set_ylabel('x', fontsize=14)
    ax_bif.set_title('Bifurcation Diagram', fontsize=16, fontweight='bold')
    
    # Mark selected r values
    for r in r_values:
        ax_bif.axvline(x=r, color='red', linestyle='--', alpha=0.5)
    
    # Time series and cobweb for each r value
    for idx, r in enumerate(r_values):
        # Time series
        ax_ts = fig.add_subplot(gs[idx, 1])
        x_vals = [x0]
        x = x0
        for _ in range(iterations):
            x = logistic_map(x, r)
            x_vals.append(x)
        
        ax_ts.plot(range(iterations+1), x_vals, 'b-', linewidth=1.5, alpha=0.7)
        ax_ts.scatter(range(iterations+1), x_vals, c=x_vals, cmap='plasma', s=10, alpha=0.8)
        ax_ts.set_xlabel('n', fontsize=12)
        ax_ts.set_ylabel('x_n', fontsize=12)
        ax_ts.set_title(f'r = {r:.2f}', fontsize=14)
        ax_ts.grid(True, alpha=0.3)
        ax_ts.set_ylim(0, 1)
        
        # Cobweb plot
        ax_cw = fig.add_subplot(gs[idx, 2])
        x_line = np.linspace(0, 1, 500)
        ax_cw.plot(x_line, logistic_map(x_line, r), 'b-', linewidth=2)
        ax_cw.plot(x_line, x_line, 'k--', alpha=0.5)
        
        xn = x0
        for _ in range(20):  # Show fewer iterations for clarity
            x_next = logistic_map(xn, r)
            ax_cw.plot([xn, xn], [xn, x_next], 'r-', linewidth=1, alpha=0.5)
            ax_cw.plot([xn, x_next], [x_next, x_next], 'r-', linewidth=1, alpha=0.5)
            xn = x_next
        
        ax_cw.set_xlim(0, 1)
        ax_cw.set_ylim(0, 1)
        ax_cw.set_xlabel('x_n', fontsize=12)
        ax_cw.set_ylabel('x_{n+1}', fontsize=12)
        ax_cw.grid(True, alpha=0.3)
        ax_cw.set_aspect('equal')
    
    plt.suptitle('Comprehensive Logistic Map Analysis', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save:
        plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_period_doubling_route(save=False):
    """Illustrate the period-doubling route to chaos"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    r_values = [2.8, 3.2, 3.5, 3.55, 3.57, 3.83]
    titles = ['Period 1', 'Period 2', 'Period 4', 'Period 8', 'Chaos', 'Period 3 Window']
    
    for idx, (r, title) in enumerate(zip(r_values, titles)):
        ax = axes[idx]
        
        # Generate time series
        x = 0.2
        x_vals = [x]
        for _ in range(200):
            x = logistic_map(x, r)
            x_vals.append(x)
        
        # Plot last 50 points
        n = range(150, 201)
        ax.plot(n, x_vals[150:201], 'b-', linewidth=2, alpha=0.7)
        ax.scatter(n, x_vals[150:201], c='red', s=30, zorder=5)
        
        ax.set_xlabel('n', fontsize=12)
        ax.set_ylabel('x_n', fontsize=12)
        ax.set_title(f'{title}\nr = {r:.2f}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.suptitle('Period-Doubling Route to Chaos', fontsize=18, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig('period_doubling.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == "__main__":    
    # Example 1: Cobweb plot for a periodic orbit
    print("\n1. Generating Cobweb Plot (r=3.2, periodic)...")
    plot_cobweb(r=3.2, x0=0.2, iterations=50)
    
    # Example 2: Cobweb plot for chaos
    print("\n2. Generating Cobweb Plot (r=3.9, chaotic)...")
    plot_cobweb(r=3.9, x0=0.2, iterations=100)
    
    # Example 3: Time series for different regimes
    print("\n3. Generating Time Series (periodic)...")
    plot_time_series(r=3.2, x0=0.2, iterations=100)
    
    print("\n4. Generating Time Series (chaotic)...")
    plot_time_series(r=3.9, x0=0.2, iterations=100)
    
    # Example 4: Bifurcation diagram
    print("\n5. Generating Bifurcation Diagram (this may take a moment)...")
    plot_bifurcation()
    
    # Example 5: Lyapunov exponent
    print("\n6. Generating Lyapunov Exponent Plot...")
    plot_lyapunov()
    
    # # Example 6: Comprehensive analysis
    # print("\n7. Generating Comprehensive Analysis...")
    # plot_comprehensive_analysis()
    
    # # Example 7: Period-doubling route
    # print("\n8. Generating Period-Doubling Illustration...")
    # plot_period_doubling_route()
    
    print("\nAll plots generated successfully!")