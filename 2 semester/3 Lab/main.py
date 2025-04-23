import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import time
import math
import json
import random

class TSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Муравьиный алгоритм")
        self.root.state('zoomed')
        
        # Основной фрейм
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Левая часть
        left_frame = tk.Frame(main_frame, width=200)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        left_frame.pack_propagate(False)
        
        # Таблица рёбер
        self.table = ttk.Treeview(left_frame, columns=("Note1", "Note2", "Weight"), show="headings")
        self.table.heading("Note1", text="Вершина 1")
        self.table.heading("Note2", text="Вершина 2")
        self.table.heading("Weight", text="Длина")
        self.table.column("Note1", width=60, anchor=tk.CENTER)
        self.table.column("Note2", width=60, anchor=tk.CENTER)
        self.table.column("Weight", width=60, anchor=tk.CENTER)
        self.table.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Поле результата
        self.result_text = tk.Text(left_frame, height=6)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Модификация
        self.use_optimization = tk.BooleanVar()
        self.opt_checkbox = tk.Checkbutton(left_frame, text="Использовать модификацию", 
                                          variable=self.use_optimization)
        self.opt_checkbox.pack(padx=5, pady=5)
        
        # Параметры алгоритма
        param_frame = tk.Frame(left_frame)
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(param_frame, text="Количество муравьев:").pack()
        self.n_ants = tk.Entry(param_frame)
        self.n_ants.insert(0, "10")
        self.n_ants.pack(fill=tk.X)
        
        tk.Label(param_frame, text="Количество итераций:").pack()
        self.iterations = tk.Entry(param_frame)
        self.iterations.insert(0, "100")
        self.iterations.pack(fill=tk.X)

        tk.Label(param_frame, text="Коэффициент испарения:").pack()
        self.evaporation = tk.Entry(param_frame)
        self.evaporation.insert(0, "0.5")
        self.evaporation.pack(fill=tk.X)
        
        tk.Label(param_frame, text="Альфа (влияние феромонов):").pack()
        self.alpha = tk.Entry(param_frame)
        self.alpha.insert(0, "1")
        self.alpha.pack(fill=tk.X)
        
        tk.Label(param_frame, text="Бета (влияние расстояния):").pack()
        self.beta = tk.Entry(param_frame)
        self.beta.insert(0, "2")
        self.beta.pack(fill=tk.X)
        
        # Кнопки
        button_frame = tk.Frame(left_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.solve_btn = tk.Button(button_frame, text="Рассчитать", command=self.solve_tsp)
        self.solve_btn.pack(fill=tk.X)
        
        self.undo_btn = tk.Button(button_frame, text="Отмена", command=self.undo_action)
        self.undo_btn.pack(fill=tk.X)
        
        self.clear_btn = tk.Button(button_frame, text="Очистить", command=self.clear_graph)
        self.clear_btn.pack(fill=tk.X)

        self.load_btn = tk.Button(button_frame, text="Загрузить граф", command=self.load_graph)
        self.load_btn.pack(fill=tk.X)
        
        # Правая часть
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.graph_canvas = tk.Canvas(right_frame, bg="white")
        self.graph_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.solution_canvas = tk.Canvas(right_frame, bg="white")
        self.solution_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.nodes = {}
        self.edges = []
        self.history = []
        self.selected_node = None
        
        self.graph_canvas.bind("<Button-1>", self.add_node_or_edge)
    
    def add_node_or_edge(self, event):
        pos = (event.x, event.y)
        clicked_node = self.find_clicked_node(pos)
        
        if clicked_node is None:
            self.add_node(pos)
        elif self.selected_node is None:
            self.selected_node = clicked_node
        else:
            if self.selected_node != clicked_node:
                self.add_edge(self.selected_node, clicked_node)
            self.selected_node = None
    
    def add_node(self, pos):
        node_id = len(self.nodes)
        self.nodes[node_id] = pos
        self.history.append(("node", node_id))
        self.redraw_graph()
    
    def add_edge(self, node1, node2):
        x1, y1 = self.nodes[node1]
        x2, y2 = self.nodes[node2]
        weight = round(np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])), 2)
        
        self.edges.append([node1, node2, weight])
        self.history.append(("edge", node1, node2))
        self.redraw_graph()
    
    def find_clicked_node(self, pos):
        for node, (x, y) in self.nodes.items():
            if (x - 15 <= pos[0] <= x + 15) and (y - 15 <= pos[1] <= y + 15):
                return node
        return None
    
    def redraw_graph(self):
        self.graph_canvas.delete("all")
        self.table.delete(*self.table.get_children())

        for node, (x, y) in self.nodes.items():
            self.graph_canvas.create_oval(x-10, y-10, x+10, y+10, fill="red")
            self.graph_canvas.create_text(x, y, text=str(node), fill="white")

        for node1, node2, weight in self.edges:
            x1, y1 = self.nodes[node1]
            x2, y2 = self.nodes[node2]
            
            angle = math.atan2(y2 - y1, x2 - x1)
            
            start_x = x1 + 10 * math.cos(angle)
            start_y = y1 + 10 * math.sin(angle)
            
            end_x = x2 - 10 * math.cos(angle)
            end_y = y2 - 10 * math.sin(angle)

            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            offset = 15
            text_x = mid_x + offset * math.sin(angle)
            text_y = mid_y - offset * math.cos(angle)
                        
            self.graph_canvas.create_line(start_x, start_y, end_x, end_y, arrow=tk.LAST, fill="blue", width=4)
            self.graph_canvas.create_text(text_x, text_y, text=str(weight), fill="black")
            
            self.table.insert("", tk.END, values=(node1, node2, weight))

    def get_distance_matrix(self):
        n = len(self.nodes)
        dist_matrix = np.full((n, n), float('inf'))
        
        for edge in self.edges:
            i, j, weight = edge
            dist_matrix[i][j] = weight
        
        return dist_matrix

    def calculate_path_length(self, path, dist_matrix):
        length = 0
        for i in range(len(path)-1):
            length += dist_matrix[path[i]][path[i+1]]
        length += dist_matrix[path[-1]][path[0]]
        return length

    def solve_tsp(self):
        if not self.edges:
            messagebox.showwarning("Ошибка", "Добавьте рёбра в граф!")
            return
        
        try:
            n_ants = int(self.n_ants.get())
            max_iter = int(self.iterations.get())
            evaporation = float(self.evaporation.get())
            alpha = float(self.alpha.get())
            beta = float(self.beta.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные параметры алгоритма!")
            return
        
        dist_matrix = self.get_distance_matrix()
        n = len(self.nodes)
        
        pheromone = np.ones((n, n)) / n
        
        best_path = None
        best_length = float('inf')
        
        start_time = time.time()
        
        for iteration in range(max_iter):
            if self.use_optimization.get():
                start_city = random.randint(0, n-1)
                paths = self.generate_ant_paths(start_city, n_ants, dist_matrix, pheromone, alpha, beta)
            else:
                paths = []
                for ant in range(n_ants):
                    start_city = random.randint(0, n-1)
                    path = self.generate_ant_path(start_city, dist_matrix, pheromone, alpha, beta)
                    path_length = self.calculate_path_length(path, dist_matrix)
                    paths.append((path, path_length))
            
            pheromone *= (1 - evaporation)

            current_best_path, current_best_length = min(paths, key=lambda x: x[1])
            if current_best_length < best_length:
                best_path = current_best_path
                best_length = current_best_length
            
            for i in range(len(current_best_path)):
                j = (i + 1) % len(current_best_path)
                pheromone[current_best_path[i]][current_best_path[j]] += 1.0 / current_best_length
        
        end_time = time.time()
        
        if best_path is not None:
            best_path.append(best_path[0])
            self.visualize_solution(best_path)
            method_name = "Блуждающая колония" if self.use_optimization.get() else "Обычный муравьиный алгоритм"
            self.result_text.insert(tk.END, 
                f"Метод: {method_name}\n"
                f"Лучший путь: {' -> '.join(map(str, best_path))}\n"
                f"Длина: {best_length:.2f}\n"
                f"Время: {end_time - start_time:.2f} сек\n")
        else:
            self.result_text.insert(tk.END, "Полный цикл не найден!\n")
    
    def generate_ant_paths(self, start_city, n_ants, dist_matrix, pheromone, alpha, beta):
        paths = []
        n = len(dist_matrix)
        
        for ant in range(n_ants):
            path = [start_city]
            visited = set([start_city])
            
            while len(visited) < n:
                current_city = path[-1]
                next_city = self.select_next_city(current_city, visited, dist_matrix, pheromone, alpha, beta)
                path.append(next_city)
                visited.add(next_city)
            
            path_length = self.calculate_path_length(path, dist_matrix)
            paths.append((path, path_length))
        
        return paths
    
    def generate_ant_path(self, start_city, dist_matrix, pheromone, alpha, beta):
        path = [start_city]
        visited = set([start_city])
        n = len(dist_matrix)
        
        while len(visited) < n:
            current_city = path[-1]
            next_city = self.select_next_city(current_city, visited, dist_matrix, pheromone, alpha, beta)
            path.append(next_city)
            visited.add(next_city)
        
        return path
    
    def select_next_city(self, current_city, visited, dist_matrix, pheromone, alpha, beta):
        unvisited = [city for city in range(len(dist_matrix)) if city not in visited]
        probabilities = []
        
        total = 0.0
        for city in unvisited:
            if dist_matrix[current_city][city] > 0:
                pheromone_factor = pheromone[current_city][city] ** alpha
                heuristic_factor = (1.0 / dist_matrix[current_city][city]) ** beta
                total += pheromone_factor * heuristic_factor
        
        probabilities = []
        for city in unvisited:
            if dist_matrix[current_city][city] > 0:
                pheromone_factor = pheromone[current_city][city] ** alpha
                heuristic_factor = (1.0 / dist_matrix[current_city][city]) ** beta
                probability = (pheromone_factor * heuristic_factor) / total
                probabilities.append((city, probability))
            else:
                probabilities.append((city, 0))
        
        sum_probs = sum(p for c, p in probabilities)
        if sum_probs > 0:
            probabilities = [(c, p/sum_probs) for c, p in probabilities]
        else:
            return random.choice(unvisited)
        
        r = random.random()
        cumulative = 0.0
        for city, probability in probabilities:
            cumulative += probability
            if r <= cumulative:
                return city
        
        return unvisited[0]
    
    def visualize_solution(self, path):
        self.solution_canvas.delete("all")
        
        for node, (x, y) in self.nodes.items():
            self.solution_canvas.create_oval(x-10, y-10, x+10, y+10, fill="red")
            self.solution_canvas.create_text(x, y, text=str(node), fill="white")
        
        for i in range(len(path)-1):
            node1, node2 = path[i], path[i+1]
            x1, y1 = self.nodes[node1]
            x2, y2 = self.nodes[node2]
            
            angle = math.atan2(y2 - y1, x2 - x1)        
            start_x = x1 + 10 * math.cos(angle)
            start_y = y1 + 10 * math.sin(angle)
            
            end_x = x2 - 10 * math.cos(angle)
            end_y = y2 - 10 * math.sin(angle)
                        
            self.solution_canvas.create_line(start_x, start_y, end_x, end_y, arrow=tk.LAST, fill="blue", width=4)

    def load_graph(self):
        filepath = filedialog.askopenfilename(
            title="Выберите файл графа",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )

        if not filepath:
            return

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.clear_graph()

            for node_id, pos in data["nodes"].items():
                self.nodes[int(node_id)] = tuple(pos)

            for edge in data["edges"]:
                self.edges.append([edge[0], edge[1], edge[2]])

            self.redraw_graph()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{str(e)}")
    
    def get_distance(self, node1, node2):
        for edge in self.edges:
            if (edge[0], edge[1]) == (node1, node2):
                return edge[2]
        return float("inf")
    
    def undo_action(self):
        if not self.history:
            return
        action = self.history.pop()
        if action[0] == "node":
            del self.nodes[action[1]]
            self.edges = [e for e in self.edges if e[0] != action[1] and e[1] != action[1]]
        elif action[0] == "edge":
            self.edges.remove([action[1], action[2], self.get_distance(action[1], action[2])])
        self.redraw_graph()
    
    def clear_graph(self):
        self.nodes.clear()
        self.edges.clear()
        self.history.clear()
        self.redraw_graph()
        self.solution_canvas.delete("all")
        self.result_text.delete(1.0, tk.END)

root = tk.Tk()
app = TSPApp(root)
root.mainloop()
