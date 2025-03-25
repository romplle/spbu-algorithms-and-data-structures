import random
import tkinter as tk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

swarm = None
velocities = None
personal_best_positions = None
personal_best_values = None
global_best_position = None
global_best_value = float('inf')
total_iterations = 0

min_position = -6.0
max_position = 6.0

def function(x1, x2):
    return x1**3 + x2**2 - 3 * x1 - 2 * x2 + 2

def initialize_particles(number_of_particles):
    global swarm, velocities, personal_best_positions, personal_best_values, global_best_position, global_best_value
    swarm = np.random.uniform(min_position, max_position, (number_of_particles, 2))
    velocities = np.random.uniform(-1, 1, (number_of_particles, 2))
    
    personal_best_positions = np.copy(swarm)
    personal_best_values = np.array([function(x1, x2) for x1, x2 in swarm])
    
    best_particle_idx = np.argmin(personal_best_values)
    global_best_position = personal_best_positions[best_particle_idx]
    global_best_value = personal_best_values[best_particle_idx]

def update_particles(inertion, lbest_coefficient, gbest_coefficient):
    global swarm, velocities, personal_best_positions, personal_best_values, global_best_position, global_best_value
    
    for i in range(len(swarm)):
        random1, random2 = random.random(), random.random()
        velocities[i] = (
            inertion * velocities[i] +
            lbest_coefficient * random1 * (personal_best_positions[i] - swarm[i]) +
            gbest_coefficient * random2 * (global_best_position - swarm[i])
        )
        swarm[i] += velocities[i]
        
        swarm[i] = np.clip(swarm[i], min_position, max_position)
        
        current_fitness = function(swarm[i][0], swarm[i][1])
        if current_fitness < personal_best_values[i]:
            personal_best_positions[i] = swarm[i]
            personal_best_values[i] = current_fitness
        
        if current_fitness < global_best_value:
            global_best_position = swarm[i]
            global_best_value = current_fitness

def plot_particles():
    ax.clear()
    ax.set_xlim(min_position, max_position)
    ax.set_ylim(min_position, max_position)
    ax.grid(True, zorder=1)
    
    ax.axhline(0, color="black", linewidth=1, zorder=2)
    ax.axvline(0, color="black", linewidth=1, zorder=2)
    
    x_vals, y_vals = swarm[:, 0], swarm[:, 1]
    ax.scatter(x_vals, y_vals, color="blue", s=10, label="Частицы", zorder=4)
    ax.scatter(global_best_position[0], global_best_position[1], color="red", marker="*", s=50, label="Global Best", zorder=3)
    
    ax.legend(fontsize=10)
    canvas.draw()

def run_algorithm():
    global total_iterations
    number_of_particles = int(particles_input.get())
    number_of_iterations = int(iterations_input.get())
    
    if swarm is None:
        initialize_particles(number_of_particles)
    
    inertion = float(inertion_input.get())
    lbest_coefficient = float(lbest_coefficient_input.get())
    gbest_coefficient = float(gbest_coefficient_input.get())

    for _ in range(number_of_iterations):
        update_particles(inertion, lbest_coefficient, gbest_coefficient)
        total_iterations += 1

    plot_particles()
    iterations_count_label.config(text=f"Общее количество итераций: {total_iterations}")
    result_label.config(text=f"Лучшее решение:\nX[1] = {global_best_position[0]:.5f}\nX[2] = {global_best_position[1]:.5f}")
    function_value_label.config(text=f"Значение функции: {global_best_value:.5f}")

# Интерфейс
root = tk.Tk()
root.title("Алгоритм роя частиц")
root.geometry("1000x650")

# Параметры
params_frame = tk.LabelFrame(root, text="Параметры", padx=10, pady=10)
params_frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")

tk.Label(params_frame, text="Коэфф. текущей скорости:").grid(row=1, column=0, sticky="w")
inertion_input = tk.Entry(params_frame)
inertion_input.insert(0, "0.3")
inertion_input.grid(row=1, column=1)

tk.Label(params_frame, text="Коэфф. собственного лучшего значения:").grid(row=2, column=0, sticky="w")
lbest_coefficient_input = tk.Entry(params_frame)
lbest_coefficient_input.insert(0, "2")
lbest_coefficient_input.grid(row=2, column=1)

tk.Label(params_frame, text="Коэфф. глобального лучшего значения:").grid(row=3, column=0, sticky="w")
gbest_coefficient_input = tk.Entry(params_frame)
gbest_coefficient_input.insert(0, "3")
gbest_coefficient_input.grid(row=3, column=1)

tk.Label(params_frame, text="Количество частиц:").grid(row=4, column=0, sticky="w")
particles_input = tk.Entry(params_frame)
particles_input.insert(0, "30")
particles_input.grid(row=4, column=1)

# Управление
controls_frame = tk.LabelFrame(root, text="Управление", padx=10, pady=10)
controls_frame.grid(row=1, column=0, padx=10, pady=10, sticky="w")

tk.Label(controls_frame, text="Количество итераций:").grid(row=0, column=0, sticky="w")
iterations_input = tk.Entry(controls_frame)
iterations_input.insert(0, "1")
iterations_input.grid(row=0, column=1)

run_button = tk.Button(controls_frame, text="Рассчитать", command=run_algorithm)
run_button.grid(row=2, column=0, columnspan=2, pady=5)

iterations_count_label = tk.Label(controls_frame, text="Общее количество итераций:")
iterations_count_label.grid(row=4, column=0, sticky="w")

# Результаты
results_frame = tk.LabelFrame(root, text="Результаты", padx=10, pady=10)
results_frame.grid(row=2, column=0, padx=10, pady=10, sticky="w")

result_label = tk.Label(results_frame, text="Лучшее решение:")
result_label.grid(row=0, column=0, sticky="w")

function_value_label = tk.Label(results_frame, text="Значение функции:")
function_value_label.grid(row=1, column=0, sticky="w")

# График
fig = Figure(figsize=(5, 5), dpi=100)
ax = fig.add_subplot(111)
ax.set_title("Позиции частиц")
ax.set_xlim(min_position, max_position)
ax.set_ylim(min_position, max_position)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().grid(row=0, column=1, rowspan=3, padx=10, pady=10)

# Запуск интерфейса
root.mainloop()
