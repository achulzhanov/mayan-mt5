import pandas as pd
import matplotlib.pyplot as plt
import csv

# --- Configuration ---
FILE_PATH = 'training_logs.csv'  # Change this if your file is in a different folder

def parse_logs(file_path):
    training_data = []
    eval_data = []

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip comments and empty lines
            if not row or row[0].startswith('#'):
                continue
            
            # Convert all values to float
            try:
                values = [float(x) for x in row]
            except ValueError:
                continue # Skip lines that can't be parsed

            # SEPARATE BASED ON COLUMN COUNT
            # Training rows have 5 columns
            if len(values) == 5:
                training_data.append(values)
            # Evaluation rows have 9 columns
            elif len(values) == 9:
                eval_data.append(values)

    # Create DataFrames
    df_train = pd.DataFrame(training_data, columns=[
        'Train_Loss', 'Grad_Norm', 'Learning_Rate', 'Epoch', 'Step'
    ])
    
    df_eval = pd.DataFrame(eval_data, columns=[
        'Eval_Loss', 'BLEU', 'chrF', 'TER', 
        'Runtime', 'Samps_Per_Sec', 'Steps_Per_Sec', 'Epoch', 'Step'
    ])

    return df_train, df_eval

def plot_data(df_train, df_eval):
    # Setup the plot style
    plt.style.use('ggplot')
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Training & Evaluation Metrics', fontsize=16)

    # 1. LOSS (Training vs Evaluation)
    ax = axes[0, 0]
    ax.plot(df_train['Step'], df_train['Train_Loss'], label='Train Loss', alpha=0.6, linewidth=1)
    ax.plot(df_eval['Step'], df_eval['Eval_Loss'], label='Eval Loss', color='red', linewidth=2)
    ax.set_title('Loss Curves')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)

    # 2. TRANSLATION QUALITY (BLEU & chrF)
    ax = axes[0, 1]
    ax.plot(df_eval['Step'], df_eval['BLEU'], label='BLEU', color='blue', marker='x')
    ax.plot(df_eval['Step'], df_eval['chrF'], label='chrF', color='green', marker='.')
    ax.set_title('Translation Quality Metrics')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True)

    # 3. GRADIENT NORM
    ax = axes[1, 0]
    # Filter out extreme initial spikes for better visibility if needed
    ax.plot(df_train['Step'], df_train['Grad_Norm'], color='purple', alpha=0.7)
    ax.set_title('Gradient Norm')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Norm')
    ax.grid(True)

    # 4. LEARNING RATE
    ax = axes[1, 1]
    ax.plot(df_train['Step'], df_train['Learning_Rate'], color='orange')
    ax.set_title('Learning Rate Schedule')
    ax.set_xlabel('Steps')
    ax.set_ylabel('LR')
    ax.grid(True)

    # 5. TER (Translation Edit Rate) - Lower is better
    ax = axes[2, 0]
    ax.plot(df_eval['Step'], df_eval['TER'], color='brown', marker='v')
    ax.set_title('TER (Lower is Better)')
    ax.set_xlabel('Steps')
    ax.set_ylabel('TER Score')
    ax.grid(True)

    # 6. EPOCH PROGRESS
    ax = axes[2, 1]
    ax.plot(df_train['Step'], df_train['Epoch'], color='black', linestyle='--')
    ax.set_title('Training Progress')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Epochs')
    ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    try:
        print("Parsing CSV...")
        train_df, eval_df = parse_logs(FILE_PATH)
        
        if not train_df.empty:
            print(f"Loaded {len(train_df)} training rows.")
            print(f"Loaded {len(eval_df)} evaluation rows.")
            print("Plotting...")
            plot_data(train_df, eval_df)
        else:
            print("Error: No data found in CSV.")
            
    except FileNotFoundError:
        print(f"Error: Could not find file '{FILE_PATH}'. Make sure it is in the same folder.")