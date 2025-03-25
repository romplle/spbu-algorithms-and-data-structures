import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import time
import math
import json

class TSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Метод ближайшего соседа")
        self.root.geometry("1400x800")
        
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

    def solve_tsp(self):
        if not self.edges:
            messagebox.showwarning("Ошибка", "Добавьте рёбра в граф!")
            return
        
        start_time = time.perf_counter()
        best_path = None
        best_distance = float("inf")
        start_nodes = self.nodes.keys() if self.use_optimization.get() else [next(iter(self.nodes))]
        
        for start in start_nodes:
            path = [start]
            unvisited = set(self.nodes) - {start}
            total_distance = 0
            
            while unvisited:
                nearest = min(unvisited, key=lambda x: self.get_distance(path[-1], x))
                total_distance += self.get_distance(path[-1], nearest)
                path.append(nearest)
                unvisited.remove(nearest)
            
            if self.get_distance(path[-1], start) != float('inf'):
                total_distance += self.get_distance(path[-1], start)
                path.append(start)
            else:
                total_distance = float('inf')
            
            if total_distance < best_distance:
                best_distance = total_distance
                best_path = path
        
        end_time = time.perf_counter()
        
        if best_path:
            self.visualize_solution(best_path)
            self.result_text.insert(tk.END, f"Лучший путь: {' -> '.join(map(str, best_path))}\nДлина: {best_distance:.4f}\nВремя: {end_time - start_time:.4f} сек\n")
        else:
            self.result_text.insert(tk.END, "Полный цикл не найден!\n")
    
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
            messagebox.showinfo("Успех", "Граф загружен!")

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
