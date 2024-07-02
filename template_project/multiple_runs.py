# multiple_runs.py
import subprocess
import numpy as np

def run_experiment(num_runs):
    accuracies = []
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}")
        result = subprocess.run(['python', 'run.py'], capture_output=True, text=True)
        output = result.stdout
        # Extract the test accuracy from the output
        for line in output.split('\n'):
            if "Test Accuracy:" in line:
                accuracy = float(line.split(': ')[1].strip('%'))
                accuracies.append(accuracy)
                break

    return accuracies

def main():
    num_runs = 5  # Number of runs to perform
    accuracies = run_experiment(num_runs)
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    print(f"Test Accuracies: {accuracies}")
    print(f"Mean Test Accuracy: {mean_accuracy:.2f}%")
    print(f"Standard Deviation of Test Accuracy: {std_accuracy:.2f}%")

if __name__ == '__main__':
    main()