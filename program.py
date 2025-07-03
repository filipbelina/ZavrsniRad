import os
import tkinter as tk
from trainerFullRun import train_full_run
from runAiSimulation import run_ai_simulation
from Trainer import EvolutionaryTrainer
from trainerMultiMove import train_multi_move
from manual_play import manual_play

CONFIG_FILE = 'config.txt'

def load_config():
    config = {
        "population_size": 100,
        "generations": 50,
        "mutation_rate": 0.3,
        "fitness_function": 4,
        "training_algorithm": 2,
        "evaluations": 5000,
        "neural_network": "neural_networks/final_model.pkl",
        "seed": 1
    }
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            for line in f:
                key, value = line.strip().split('=')
                config[key] = value if key == "neural_network" else (float(value) if '.' in value else int(value))
    return config

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}={value}\n")

def open_config_editor():
    config = load_config()

    def save_changes():
        for key in entries:
            if key == "neural_network":
                config[key] = f"neural_networks/{entries[key].get()}"
            else:
                config[key] = int(entries[key].get()) if key != "mutation_rate" else float(entries[key].get())
        save_config(config)
        editor.destroy()

    editor = tk.Toplevel(root)
    editor.title("Edit Configuration")
    editor.geometry("400x400")
    editor.configure(bg="#334E58")

    title_label = tk.Label(editor, text="Edit Configuration", font=("Arial", 18, "bold"), bg="#334E58", fg="white")
    title_label.pack(pady=10)

    entries_frame = tk.Frame(editor, bg="#334E58")
    entries_frame.pack(expand=True, padx=20, pady=10)

    entries = {}
    for i, (key, value) in enumerate(config.items()):
        tk.Label(entries_frame, text=key.replace('_', ' ').capitalize(), font=("Arial", 12), bg="#334E58", fg="white").grid(row=i, column=0, padx=10, pady=5, sticky="w")
        entry = tk.Entry(entries_frame, font=("Arial", 12), bg="#6B6D76", fg="white")
        entry.insert(0, str(value) if key != "neural_network" else value.split('/')[-1])
        entry.grid(row=i, column=1, padx=10, pady=5)
        entries[key] = entry

    save_button = tk.Button(editor, text="Save", command=save_changes, width=20, height=2, bg="#6B6D76", fg="white", font=("Arial", 12))
    save_button.pack(pady=10)

def run_training():
    os.makedirs('training_results', exist_ok=True)
    os.makedirs('best_per_gen', exist_ok=True)

    config = load_config()
    trainer = EvolutionaryTrainer(
        config["population_size"],
        config["generations"],
        config["mutation_rate"],
        config["fitness_function"],
        config["training_algorithm"],
        config["evaluations"]
    )

    if trainer.training_algorithm in [1, 3]:
        train_full_run(trainer)
    elif trainer.training_algorithm in [2, 4]:
        train_multi_move(trainer)
    else:
        print("Invalid training algorithm selected.")
        return

    stats_file = f'training_results/stats.txt'
    best_model_file = f'training_results/best_model.pkl'

    if os.path.exists('generation_stats.txt'):
        os.rename('generation_stats.txt', stats_file)
    if os.path.exists('best_nn.pkl'):
        os.rename('best_nn.pkl', best_model_file)

def run_simulation():
    print("Starting AI simulation...")
    config = load_config()
    run_ai_simulation(config["neural_network"], config["seed"])

def exit_program():
    print("Exiting program. Goodbye!")
    root.destroy()

def manual_play_game():
    print("Starting manual play...")
    manual_play()

root = tk.Tk()
root.title("Tetris")

root.geometry("400x400")
root.configure(bg="#334E58")

title_frame = tk.Frame(root, bg="#334E58")
title_frame.pack(fill="x", pady=10)

title_label = tk.Label(title_frame, text="Tetris", font=("Arial", 18, "bold"), bg="#334E58", fg="white")
title_label.pack()

button_frame = tk.Frame(root, bg="#334E58")
button_frame.pack(expand=True)

config_button = tk.Button(button_frame, text="Play Yourself", command=manual_play_game, width=20, height=2, bg="#6B6D76", fg="white", font=("Arial", 12))
config_button.pack(pady=10)

train_button = tk.Button(button_frame, text="Run Training", command=run_training, width=20, height=2, bg="#6B6D76", fg="white", font=("Arial", 12))
train_button.pack(pady=10)

simulation_button = tk.Button(button_frame, text="Run AI Simulation", command=run_simulation, width=20, height=2, bg="#6B6D76", fg="white", font=("Arial", 12))
simulation_button.pack(pady=10)

config_button = tk.Button(button_frame, text="Edit Configuration", command=open_config_editor, width=20, height=2, bg="#6B6D76", fg="white", font=("Arial", 12))
config_button.pack(pady=10)

exit_button = tk.Button(button_frame, text="Exit", command=exit_program, width=20, height=2, bg="#e74c3c", fg="white", font=("Arial", 12))
exit_button.pack(pady=10)

root.mainloop()