import matplotlib.pyplot as plt
import seaborn as sns


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_from_dataframe(df, metric, save_path=None):
    """
    Plots mean ± standard deviation for each method for the specified metric.
    
    df: DataFrame with columns ['method', 'metric', 'value']
    metric: str, which metric to plot
    save_path: optional path to save the figure
    """
    plt.figure(figsize=(8, 5))
    
    # Filter for the metric
    subset = df[df['metric'] == metric]
    
    # Compute mean and std for each method
    stats = subset.groupby('method')['value'].agg([np.mean, np.std]).reset_index()
    
    # Plot
    plt.errorbar(
        x=stats['method'],
        y=stats['mean'],
        yerr=stats['std'],  # error bars show ±1 standard deviation
        fmt='o',
        capsize=5,
        markersize=8,
        linestyle='None',
        ecolor='black'
    )
    
    plt.title(f"{metric.replace('_',' ').capitalize()} Comparison")
    plt.xticks(rotation=45)
    plt.ylabel(metric)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)

    plt.close()
 
