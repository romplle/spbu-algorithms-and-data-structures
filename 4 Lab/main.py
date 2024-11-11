import random
import tkinter as tk
from tkinter import ttk

def function(x1, x2):
    return x1**3 + x2**2 - 3 * x1 - 2 * x2 + 2

def create_initial_population():
    if use_real_version.get():
        return [(random.uniform(min_value, max_value), random.uniform(min_value, max_value)) for _ in range(number_of_chromosomes)]
    else:
        return [(random.randint(min_value, max_value), random.randint(min_value, max_value)) for _ in range(number_of_chromosomes)]

def fitness(individual):
    x1, x2 = individual
    if function(x1, x2) == -1:
        return float("inf")
    return 1 / (1 + function(x1, x2))

def segment_population(population):
    return [population[i:i+4] for i in range(0, len(population), 4)]

def ranking_selection(segment):
    sorted_segment = sorted(segment, key=lambda x: fitness(x), reverse=True)
    return sorted_segment[:3]

def crossover(parent1, parent2, alpha=0.5):
    child1 = alpha * parent1[0] + (1 - alpha) * parent2[0], alpha * parent1[1] + (1 - alpha) * parent2[1]
    child2 = (1 - alpha) * parent1[0] + alpha * parent2[0], (1 - alpha) * parent1[1] + alpha * parent2[1]

    return child1, child2

def mutate(individual):
    x1, x2 = individual
    if use_real_version.get():
        if random.random() < mutation_probability:
            x1 += random.gauss(0, 1)
        if random.random() < mutation_probability:
            x2 += random.gauss(0, 1)
    else: 
        if random.random() < mutation_probability:
            x1 += random.randint(-1, 1)
        if random.random() < mutation_probability:
            x2 += random.randint(-1, 1)

    x1 = max(min_value, min(x1, max_value))
    x2 = max(min_value, min(x2, max_value))
    
    return (x1, x2)

def create_new_generation(modefied_segment, standart_segment, alpha=0.5):
    if use_modified_version.get():
        best = modefied_segment[0]
        others = modefied_segment[1:]

        child1, child2 = crossover(best, others[0], alpha)
        child3, child4 = crossover(best, others[1], alpha)

        mutated_child3 = mutate(child3)
        mutated_child4 = mutate(child4)

        return [child1, child2, mutated_child3, mutated_child4]
    else:
        parents = random.sample(standart_segment, 2)
        child1, child2 = crossover(parents[0], parents[1], alpha)

        child1 = mutate(child1)
        child2 = mutate(child2)

        return [child1, child2, mutate(random.choice(standart_segment)), mutate(random.choice(standart_segment))]

def toggle_mode():
    mode_label.config(
        text="Режим: Модифицированный" if use_modified_version.get() else "Режим: Обычный"
    )

def toggle_encoding():
    encoding_label.config(
        text="Режим: Вещественный" if use_real_version.get() else "Режим: Целочисленный"
    )

def run_algorithm():
    global mutation_probability, number_of_chromosomes, min_value, max_value, number_of_generations
    global global_best_solution, global_best_fitness, total_generations, population

    mutation_probability = float(mutation_input.get()) / 100
    number_of_chromosomes = int(chromosomes_input.get())
    min_value = int(min_gene_input.get())
    max_value = int(max_gene_input.get())
    number_of_generations = int(generations_input.get())

    if population is None:
        population = create_initial_population()

    if global_best_solution is None:
        global_best_solution = None
        global_best_fitness = float('-inf')

    for _ in range(number_of_generations):
        total_generations += 1
        segments = segment_population(population)
        
        new_population = []
        for segment in segments:
            selected_individuals = ranking_selection(segment)
            new_generation = create_new_generation(selected_individuals, segment)
            new_population.extend(new_generation)

        population = new_population

        for individual in population:
            individual_fitness = fitness(individual)
            if individual_fitness > global_best_fitness:
                global_best_fitness = individual_fitness
                global_best_solution = individual
            
        for row in table.get_children():
            table.delete(row)
        for i, individual in enumerate(population):
            if use_real_version.get():
                table.insert("", "end", values=(i + 1, f"{function(*individual):.5f}", f"{individual[0]:.5f}", f"{individual[1]:.5f}"))
            else:
                table.insert("", "end", values=(i + 1, int(round(function(*individual))), int(round(individual[0])), int(round(individual[1]))))

    if use_real_version.get():
        result_label.config(text=f"Лучшее решение:\nX[1] = {global_best_solution[0]:.5f}\nX[2] = {global_best_solution[1]:.5f}")
        function_value_label.config(text=f"Значение функции: {function(*global_best_solution):.5f}")
    else:
        result_label.config(text=f"Лучшее решение:\nX[1] = {int(round(global_best_solution[0]))}\nX[2] = {int(global_best_solution[1])}")
        function_value_label.config(text=f"Значение функции: {int(round(function(*global_best_solution)))}")

    generation_count_label.config(text=f"Общее количество поколений: {total_generations}")

    toggle_mode()
    toggle_encoding()

root = tk.Tk()
root.title("Генетический алгоритм")
root.geometry("850x650")

global_best_solution = None
global_best_fitness = float('-inf')
total_generations = 0
population = None

use_modified_version = tk.BooleanVar(value=True)
use_real_version = tk.BooleanVar(value=True)

# Параметры
params_frame = tk.LabelFrame(root, text="Параметры", padx=10, pady=10)
params_frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")

tk.Label(params_frame, text="Вероятность мутации, %:").grid(row=0, column=0, sticky="w")
mutation_input = tk.Entry(params_frame)
mutation_input.insert(0, "20")
mutation_input.grid(row=0, column=1)

tk.Label(params_frame, text="Количество хромосом:").grid(row=1, column=0, sticky="w")
chromosomes_input = tk.Entry(params_frame)
chromosomes_input.insert(0, "64")
chromosomes_input.grid(row=1, column=1)

tk.Label(params_frame, text="Минимальное значение гена:").grid(row=2, column=0, sticky="w")
min_gene_input = tk.Entry(params_frame)
min_gene_input.insert(0, "-50")
min_gene_input.grid(row=2, column=1)

tk.Label(params_frame, text="Максимальное значение гена:").grid(row=3, column=0, sticky="w")
max_gene_input = tk.Entry(params_frame)
max_gene_input.insert(0, "50")
max_gene_input.grid(row=3, column=1)

# Управление
controls_frame = tk.LabelFrame(root, text="Управление", padx=10, pady=10)
controls_frame.grid(row=1, column=0, padx=10, pady=10, sticky="w")

tk.Label(controls_frame, text="Количество поколений:").grid(row=0, column=0, sticky="w")
generations_input = tk.Entry(controls_frame)
generations_input.insert(0, "10")
generations_input.grid(row=2, column=1)

def set_generations(value):
    generations_input.delete(0, tk.END)
    generations_input.insert(0, str(value))

buttons_frame = tk.Frame(controls_frame)
buttons_frame.grid(row=3, column=0, columnspan=2, pady=5)
tk.Button(buttons_frame, text="1", width=10, command=lambda: set_generations(1)).pack(side="left", padx=0)
tk.Button(buttons_frame, text="10", width=10, command=lambda: set_generations(10)).pack(side="left", padx=0)
tk.Button(buttons_frame, text="100", width=10, command=lambda: set_generations(100)).pack(side="left", padx=0)
tk.Button(buttons_frame, text="1000", width=10, command=lambda: set_generations(1000)).pack(side="left", padx=0)

run_button = tk.Button(controls_frame, text="Рассчитать", command=run_algorithm)
run_button.grid(row=4, column=0, columnspan=2, pady=5)

toggle_mode_button = tk.Checkbutton(
    controls_frame, text="Использовать модификацию", variable=use_modified_version, command=toggle_mode
)
toggle_mode_button.grid(row=5, column=0, sticky="w")

encoding_toggle_button = tk.Checkbutton(
    controls_frame, text="Использовать вещественное кодирование", variable=use_real_version, command=toggle_encoding
)
encoding_toggle_button.grid(row=6, column=0, sticky="w")

generation_count_label = tk.Label(controls_frame, text="Общее количество поколений:")
generation_count_label.grid(row=7, column=0, sticky="w")

# Результаты
results_frame = tk.LabelFrame(root, text="Результаты", padx=10, pady=10)
results_frame.grid(row=2, column=0, padx=10, pady=10, sticky="w")

result_label = tk.Label(results_frame, text="Лучшее решение:")
result_label.grid(row=0, column=0, sticky="w")

function_value_label = tk.Label(results_frame, text="Значение функции:")
function_value_label.grid(row=1, column=0, sticky="w")

mode_label = tk.Label(results_frame, text="Режим: Модифицированный")
mode_label.grid(row=2, column=0, sticky="w")

encoding_label = tk.Label(results_frame, text="Кодирование: Вещественное")
encoding_label.grid(row=5, column=0, sticky="w")

# Таблица
table_frame = tk.Frame(root)
table_frame.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nw")
table = ttk.Treeview(table_frame, columns=("Номер", "Результат", "Ген 1", "Ген 2"), show="headings")
table.heading("Номер", text="Номер")
table.heading("Результат", text="Результат")
table.heading("Ген 1", text="Ген 1")
table.heading("Ген 2", text="Ген 2")
table.column("Номер", width=50, anchor="center")
table.column("Результат", width=150, anchor="center")
table.column("Ген 1", width=125, anchor="center")
table.column("Ген 2", width=125, anchor="center")
table.pack(fill="both", expand=True)

root.mainloop()
