import pandas as pd
import matplotlib.pyplot as plt
import csv
import argparse

# --- Configuration ---
FILE_PATH = 'training_logs.csv'  

def parse_logs(file_path):
    training_data = []
    eval_data = []

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            
            try:
                values = [float(x) for x in row]
            except ValueError:
                continue 

            # Training rows have 5 columns
            if len(values) == 5:
                training_data.append(values)
            # Evaluation rows have 9 columns
            elif len(values) == 9:
                eval_data.append(values)

    df_train = pd.DataFrame(training_data, columns=[
        'Train_Loss', 'Grad_Norm', 'Learning_Rate', 'Epoch', 'Step'
    ])
    
    df_eval = pd.DataFrame(eval_data, columns=[
        'Eval_Loss', 'BLEU', 'chrF', 'TER', 
        'Runtime', 'Samps_Per_Sec', 'Steps_Per_Sec', 'Epoch', 'Step'
    ])

    return df_train, df_eval

def plot_data(df_train, df_eval, graph_title):
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(graph_title, fontsize=18, fontweight='bold', y=0.95)

    window_size = max(1, len(df_train) // 100) 
    df_train['Smoothed_Train_Loss'] = df_train['Train_Loss'].rolling(window=window_size, min_periods=1).mean()
    df_train['Smoothed_Grad_Norm'] = df_train['Grad_Norm'].rolling(window=window_size, min_periods=1).mean()

    # 1. LOSS CURVES
    ax1 = axes[0]
    ax1.plot(df_train['Step'], df_train['Train_Loss'], color='gray', alpha=0.2, label='Raw Train Loss')
    ax1.plot(df_train['Step'], df_train['Smoothed_Train_Loss'], color='#1f77b4', linewidth=2, label='Smoothed Train Loss')
    ax1.plot(df_eval['Step'], df_eval['Eval_Loss'], color='#d62728', linewidth=2.5, marker='o', markersize=4, label='Eval Loss')
    ax1.set_title('Training vs. Evaluation Loss', fontsize=14)
    ax1.set_xlabel('Global Steps')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.6)

    # 2. GRADIENT NORMS (LOG SCALE)
    ax2 = axes[1]
    ax2.plot(df_train['Step'], df_train['Grad_Norm'], color='purple', alpha=0.15, linewidth=1, label='Raw Grad Norm')
    ax2.plot(df_train['Step'], df_train['Smoothed_Grad_Norm'], color='purple', linewidth=2, label='Smoothed Grad Norm')
    
    ax2.set_title('Gradient Norms (Logarithmic Scale)', fontsize=14)
    ax2.set_xlabel('Global Steps')
    ax2.set_ylabel('Gradient Norm (Log Scale)')
    
    # APPLY LOG SCALE HERE
    ax2.set_yscale('log')

    # Adding the median value text to the legend label for extra clarity
    median_val = df_train['Grad_Norm'].median()
    ax2.axhline(y=median_val, color='black', linestyle='--', alpha=0.5, label=f'Median Norm ({median_val:.2f})')
    ax2.legend()
    # Enable gridlines for both major and minor ticks on the log scale
    ax2.grid(True, which="both", ls="-", alpha=0.4)

    plt.tight_layout(pad=3.0)
    plt.show()

if __name__ == "__main__":
    # Handle the title argument or prompt
    parser = argparse.ArgumentParser(description='Graph training logs.')
    parser.add_argument('--title', type=str, help='Set the title of the graph', default=None)
    args = parser.parse_args()

    if args.title is None:
        graph_title = input("Enter a title for the graph (or press Enter for default): ").strip()
        if not graph_title:
            graph_title = 'Model Convergence Metrics'
    else:
        graph_title = args.title

    try:
        print("Parsing CSV...")
        train_df, eval_df = parse_logs(FILE_PATH)
        
        if not train_df.empty:
            print(f"Loaded {len(train_df)} training rows.")
            print(f"Loaded {len(eval_df)} evaluation rows.")
            print("Plotting Convergence Graphs...")
            plot_data(train_df, eval_df, graph_title)
        else:
            print("Error: No data found in CSV.")
            
    except FileNotFoundError:
        print(f"Error: Could not find file '{FILE_PATH}'. Make sure it is in the same folder.")