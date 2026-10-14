from .utils import plot_from_dataframe
import pandas as pd
import os

def plot_all(result_dir,plot_dir):
    os.makedirs(plot_dir,exist_ok=True)
    eval_results_file = f"{result_dir}/eval_result.csv"
    df_metrics = pd.read_csv(eval_results_file)
    print(df_metrics["method"].unique())
    for metric in df_metrics['metric'].unique():
        
        plot_from_dataframe(df_metrics, metric, save_path=f"{plot_dir}/{metric}.png")