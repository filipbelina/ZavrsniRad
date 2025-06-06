import os
import tkinter as tk
from trainerFullRun import train_full_run
from runAiSimulation import run_ai_simulation
from Trainer import EvolutionaryTrainer
from trainerMultiMove import train_multi_move



def run_training():
    print("Starting training for 10 runs...")
    os.makedirs('training_results', exist_ok=True)  # Create a folder for results

    for run in range(3, 11):
        trainer = EvolutionaryTrainer(100, 100, 0.3, 2, 1)
        print(f"Run {run}...")
        if trainer.training_algorithm in [1, 3]:
            train_full_run(trainer)
        elif trainer.training_algorithm in [2, 4]:
            train_multi_move(trainer)
        else:
            print("Invalid training algorithm selected.")
            return

        # Save stats and best model for this run
        stats_file = f'training_results/stats_run_{run}.txt'
        best_model_file = f'training_results/best_model_run_{run}.pkl'

        # Move the generated stats and best model files to the folder
        if os.path.exists('generation_stats.txt'):
            os.rename('generation_stats.txt', stats_file)
        if os.path.exists('best_nn.pkl'):
            os.rename('best_nn.pkl', best_model_file)

        print(f"Stats saved to {stats_file} and best model saved to {best_model_file}.")

    print("Training completed for 10 runs.")

def run_simulation():
    print("Starting AI simulation...")
    run_ai_simulation()

def exit_program():
    print("Exiting program. Goodbye!")
    root.destroy()

root = tk.Tk()
root.title("Tetris AI Program")

train_button = tk.Button(root, text="Run Training", command=run_training, width=20, height=2)
train_button.pack(pady=10)

simulation_button = tk.Button(root, text="Run AI Simulation", command=run_simulation, width=20, height=2)
simulation_button.pack(pady=10)

exit_button = tk.Button(root, text="Exit", command=exit_program, width=20, height=2)
exit_button.pack(pady=10)

root.mainloop()