import tkinter as tk
from trainerFullRun import train_full_run
from runAiSimulation import run_ai_simulation
from Trainer import EvolutionaryTrainer
from trainerMultiMove import train_multi_move

trainer = EvolutionaryTrainer(800, 50, 0.5, 3, 4)


def run_training():
    print("Starting training...")
    if trainer.training_algorithm == 1:
        train_full_run(trainer)
    elif trainer.training_algorithm == 2:
        train_multi_move(trainer)
    elif trainer.training_algorithm == 3:
        train_full_run(trainer)
    elif trainer.training_algorithm == 4:
        train_multi_move(trainer)
    else:
        print("Invalid training algorithm selected.")
        return

def run_simulation():
    print("Starting AI simulation...")
    run_ai_simulation(trainer)

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