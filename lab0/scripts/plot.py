import pandas as pd
import matplotlib.pyplot as plt
import os 

os.makedirs("plots", exist_ok=True)
output_dir = "plots"

def plot_python_times(csv_path):
    # Load the CSV data
    df = pd.read_csv(csv_path,header=None, names=["type", "dim", "time_taken"])
    df.columns = ["type", "dim", "time_taken"]

    # Plotting
    plt.figure(figsize=(10, 6))
    for t in sorted(df['type'].unique()):
        subset = df[df['type'] == t]
        plt.plot(subset['dim'], subset['time_taken'], marker='o', label=f"Type {t}")

    # Customize the plot
    plt.xlabel("Size")
    plt.ylabel("Python Time(sec)")
    plt.title("Python Time vs Size ")
    plt.legend(title="Type")
    plt.grid(True)
    save_path = os.path.join(output_dir,"python_times.png")
    plt.savefig(save_path)

def plot_user_times(csv_path):
    # Load the CSV data
    df = pd.read_csv(csv_path)

    # Plotting
    plt.figure(figsize=(10, 6))
    for t in sorted(df['Type'].unique()):
        subset = df[df['Type'] == t]
        plt.plot(subset['Size'], subset['User Time'], marker='o', label=f"Type {t}")

    # Customize the plot
    plt.xlabel("Size")
    plt.ylabel("Perf User Time(sec)")
    plt.title("Perf User Time vs Size ")
    plt.legend(title="Type")
    plt.grid(True)
    save_path = os.path.join(output_dir,"perf_user_times.png")
    plt.savefig(save_path)

def plot_user_multiplication_times(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Define the mapping of types to the corresponding columns
    type_column_mapping = {
        0: 'matrixMultiplyIJK',
        1: 'matrixMultiplyIKJ',
        2: 'matrixMultiplyJIK',
        3: 'matrixMultiplyJKI',
        4: 'matrixMultiplyKIJ',
        5: 'matrixMultiplyKJI'
    }

    plt.figure(figsize=(10, 6))

    # Plot the data for each type
    for t, col in type_column_mapping.items():
        filtered_data = df[df['type'] == t]
        plt.plot(filtered_data['dim'], filtered_data[col], label=f'Type {t}', marker='o')

    # Add labels, legend, and title
    plt.xlabel('Size')
    plt.ylabel('User matrixMultiply Time(sec)')
    plt.title('User matrixMultiply Time vs Size for Each Type')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(output_dir,"perf_mult_times.png")
    plt.savefig(save_path)

def plot_cache_hit_rate(csv_path):
    # Load the CSV data
    df = pd.read_csv(csv_path)

    # Plotting
    plt.figure(figsize=(10, 6))
    for t in sorted(df['Type'].unique()):
        subset = df[df['Type'] == t]
        plt.plot(subset['Size'], subset['Cache Hit Rate'], marker='o', label=f"Type {t}")

    # Customize the plot
    plt.xlabel("Size")
    plt.ylabel("Cache Hit Rate(%)")
    plt.title("Cache Hit Rate vs Size ")
    plt.legend(title="Type")
    plt.grid(True)
    save_path = os.path.join(output_dir,"cache_hit_rates.png")
    plt.savefig(save_path)

def plot_func_times(csv_path,profiler,perc):
    # Read the CSV file
    if profiler == "perf" and perc == False:
        plt_names = ['read_matrix_plot_perf','write_matrix_plot_perf','matrix_multiply_plot_perf']
        title_ = "Perf matrixMultiply Time vs Size"
        y_label = "Time(sec)"
    elif profiler == "gprof" and perc == False:
        plt_names = ['read_matrix_plot_gprof','write_matrix_plot_gprof','matrix_multiply_plot_gprof']
        title_ = "Gprof matrixMultiply Time vs Size"
        y_label = "Time(sec)"
    elif profiler == "perf" and perc == True:
        plt_names = ['read_matrix_plot_perf_percent','write_matrix_plot_perf_percent','matrix_multiply_plot_perf_percent']
        title_ = "Perf matrixMultiply Percentage Time vs Size"
        y_label = "Time(%)"
    elif profiler == "gprof" and perc == True:
        plt_names = ['read_matrix_plot_gprof_percent','write_matrix_plot_gprof_percent','matrix_multiply_plot_gprof_percent']
        title_ = "Gprof matrixMultiply Percentage Time vs Size"
        y_label = "Time(%)"

    df = pd.read_csv(csv_path)

    # Define mappings for matrix multiplication type
    multiply_columns = {
        0: "matrixMultiplyIJK",
        1: "matrixMultiplyIKJ",
        2: "matrixMultiplyJIK",
        3: "matrixMultiplyJKI",
        4: "matrixMultiplyKIJ",
        5: "matrixMultiplyKJI",
    }

    # Function to plot a specific metric
    def plot_metric(y_column, title, ylabel, file_name):
        plt.figure(figsize=(10, 6))
        for t in df['type'].unique():
            subset = df[df['type'] == t]
            plt.plot(subset['dim'], subset[y_column], label=f'Type {t}', marker='o')

        plt.title(title)
        plt.xlabel('Size')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(output_dir,f"{file_name}.png")
        plt.savefig(save_path)
        plt.close()

    # Generate plots
    plot_metric('readMatrix', 'readMatrix Time vs Size', 'readMatrix Time',plt_names[0])
    plot_metric('writeMatrix', 'writeMatrix Time vs Size', 'writeMatrix Time', plt_names[1])

    plt.figure(figsize=(10, 6))
    for t in df['type'].unique():
        subset = df[df['type'] == t]
        plt.plot(subset['dim'], subset[multiply_columns[t]], label=f'Type {t}', marker='o')

    plt.title(title_)
    plt.xlabel('Size')
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(output_dir,plt_names[2]+".png")
    plt.savefig(save_path)
    plt.close()

def main():
    plot_python_times("python_times.csv")

    plot_user_times("cache_stats.csv")

    plot_user_multiplication_times("perf_times.csv")

    plot_cache_hit_rate("cache_stats.csv")

    plot_func_times('perf_times.csv','perf',False)
    plot_func_times('perf_times_percent.csv','perf',True)
    plot_func_times('gprof_times.csv','gprof',False)
    plot_func_times('gprof_times_percent.csv','gprof',True)

if __name__ == "__main__":
    main()