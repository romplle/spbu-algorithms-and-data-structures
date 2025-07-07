import time
import numpy as np
import pandas as pd
from tkinter import *
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from scipy.spatial.distance import cdist

class ClusterLabApp(Tk):
    def __init__(self):
        super().__init__()
        self.title("Кластеризация")
        self.geometry("1200x800")
        
        # Данные
        self.df = None
        self.best_features = None
        self.anonym_df = None
        self.compactness = {'all': None, 'selected': None, 'anonym': None, 'anonym_selected': None}
        self.axes_limits = None  # Для фиксации границ осей
        self.plot_features = [0, 1, 2]  # Фиксированные признаки для визуализации
        
        # Параметры FOREL по умолчанию
        self.radius = 2.5
        self.min_cluster_size = 10
        self.max_iter = 100
        
        # Параметры СПА по умолчанию
        self.spa_features = 5
        self.pop_size = 10
        self.generations = 50
        
        # Инициализация интерфейса
        self.create_widgets()
        self.load_data()
    
    def forel_cluster(self, X):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_samples = X_scaled.shape[0]
        labels = -np.ones(n_samples, dtype=int)
        cluster_id = 0
        
        for i in range(n_samples):
            if labels[i] != -1:
                continue
                
            center = X_scaled[i]
            iterations = 0
            while iterations < self.max_iter:
                distances = cdist([center], X_scaled).flatten()
                in_cluster = np.where((distances <= self.radius) & (labels == -1))[0]
                
                if len(in_cluster) < self.min_cluster_size:
                    break
                    
                new_center = np.mean(X_scaled[in_cluster], axis=0)
                if np.linalg.norm(new_center - center) < 1e-6:
                    break
                    
                center = new_center
                iterations += 1
                
            if len(in_cluster) >= self.min_cluster_size:
                labels[in_cluster] = cluster_id
                cluster_id += 1
                
        noise_points = np.where(labels == -1)[0]
        if len(noise_points) > 0:
            for point in noise_points:
                if np.any(labels != -1):
                    distances = cdist([X_scaled[point]], X_scaled[labels != -1]).flatten()
                    nearest = np.argmin(distances)
                    labels[point] = labels[labels != -1][nearest]
                else:
                    labels[point] = 0
                    
        return labels
    
    def compute_compactness(self, X, labels):
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1:
            return float('inf')
        
        compactness = 0.0
        valid_clusters = 0
        
        for label in unique_labels:
            cluster_points = X[labels == label]
            if len(cluster_points) < 3:
                continue
                
            centroid = np.mean(cluster_points, axis=0)
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            compactness += np.mean(distances)
            valid_clusters += 1
            
        return compactness / valid_clusters if valid_clusters > 0 else float('inf')
    
    def spa_feature_selection(self, X, n_features):
        n_total = X.shape[1]
        population = [np.random.choice(n_total, size=n_features, replace=False) 
                     for _ in range(self.pop_size)]
        best_features = None
        best_compactness = float('inf')
        
        for _ in range(self.generations):
            for features in population:
                X_sub = X[:, features]
                labels = self.forel_cluster(X_sub)
                compactness = self.compute_compactness(X_sub, labels)
                
                if compactness < best_compactness:
                    best_compactness = compactness
                    best_features = features.copy()
            
            new_population = [best_features.copy()]
            for _ in range(self.pop_size-1):
                child = best_features.copy()
                if np.random.rand() < 0.3:
                    idx = np.random.randint(n_features)
                    available = list(set(range(n_total)) - set(child))
                    if available:
                        child[idx] = np.random.choice(available)
                new_population.append(child)
            population = new_population
        
        return best_features
    
    def create_widgets(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        params_frame = ttk.LabelFrame(main_frame, text="Параметры", width=350)
        params_frame.pack(side=LEFT, fill=BOTH, expand=False)
        params_frame.pack_propagate(False)

        graph_frame = ttk.Frame(main_frame)
        graph_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        forel_frame = ttk.LabelFrame(params_frame, text="Параметры FOREL")
        forel_frame.pack(fill=X, padx=5, pady=5, ipadx=5, ipady=5)

        ttk.Label(forel_frame, text="Радиус кластера:").pack(anchor=W)
        self.radius_var = DoubleVar(value=self.radius)
        ttk.Entry(forel_frame, textvariable=self.radius_var).pack(fill=X)   

        ttk.Label(forel_frame, text="Мин. размер кластера:").pack(anchor=W)
        self.min_cluster_size_var = IntVar(value=self.min_cluster_size)
        ttk.Entry(forel_frame, textvariable=self.min_cluster_size_var).pack(fill=X)

        ttk.Label(forel_frame, text="Макс. итераций:").pack(anchor=W)
        self.max_iter_var = IntVar(value=self.max_iter)
        ttk.Entry(forel_frame, textvariable=self.max_iter_var).pack(fill=X)

        spa_frame = ttk.LabelFrame(params_frame, text="Параметры СПА")
        spa_frame.pack(fill=X, padx=5, pady=5, ipadx=5, ipady=5)

        ttk.Label(spa_frame, text="Кол-во признаков:").pack(anchor=W)
        self.spa_features_var = IntVar(value=self.spa_features)
        ttk.Entry(spa_frame, textvariable=self.spa_features_var).pack(fill=X)

        ttk.Label(spa_frame, text="Размер популяции:").pack(anchor=W)
        self.pop_size_var = IntVar(value=self.pop_size)
        ttk.Entry(spa_frame, textvariable=self.pop_size_var).pack(fill=X)

        ttk.Label(spa_frame, text="Поколений:").pack(anchor=W)
        self.generations_var = IntVar(value=self.generations)
        ttk.Entry(spa_frame, textvariable=self.generations_var).pack(fill=X)

        buttons_frame = ttk.Frame(params_frame)
        buttons_frame.pack(fill=X, padx=5, pady=10)

        buttons = [
            ("1. FOREL (все признаки)", self.run_clustering),
            ("2. Отбор признаков (СПА)", self.select_features),
            ("3. FOREL (выбранные)", self.run_selected_clustering),
            ("4. Обезличить данные", self.anonymize_data),
            ("5. FOREL (обезличенные)", self.run_anonym_clustering),
            ("6. FOREL (обезличенные + выбранные)", self.run_anonym_selected_clustering)
        ]
        
        for text, command in buttons:
            ttk.Button(buttons_frame, text=text, command=command).pack(fill=X, pady=2)

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

        result_frame = ttk.Frame(graph_frame)
        result_frame.pack(side=BOTTOM, fill=X, padx=5, pady=5)
        
        self.result_text = Text(result_frame, height=5, wrap=WORD, width=80, font=('Arial', 10))
        self.result_text.pack(fill=X, padx=5, pady=5)
        self.result_text.configure(state='disabled')
    
    def load_data(self):
        data = load_breast_cancer()
        self.df = pd.DataFrame(data.data, columns=data.feature_names)
        # Устанавливаем границы осей по исходным данным
        self.set_axes_limits(self.df.values[:, self.plot_features])
    
    def set_axes_limits(self, X):
        """Фиксирует границы осей на основе первых трех признаков"""
        padding = 0.1  # 10% отступ
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        ranges = max_vals - min_vals
        self.axes_limits = [
            (min_vals[0] - padding*ranges[0], max_vals[0] + padding*ranges[0]),
            (min_vals[1] - padding*ranges[1], max_vals[1] + padding*ranges[1]),
            (min_vals[2] - padding*ranges[2], max_vals[2] + padding*ranges[2])
        ]
    
    def update_plot(self, X, labels):
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        
        # Всегда используем фиксированные признаки для визуализации
        x = X[:, self.plot_features[0]] if X.shape[1] > self.plot_features[0] else np.zeros(X.shape[0])
        y = X[:, self.plot_features[1]] if X.shape[1] > self.plot_features[1] else np.zeros(X.shape[0])
        z = X[:, self.plot_features[2]] if X.shape[1] > self.plot_features[2] else np.zeros(X.shape[0])
        
        unique_labels = np.unique(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        
        for k, col in zip(unique_labels, colors):
            cluster_members = labels == k
            ax.scatter(
                x[cluster_members], 
                y[cluster_members], 
                z[cluster_members], 
                color=col,
                s=20,
                alpha=0.7
            )

        # Фиксируем границы осей
        if self.axes_limits:
            ax.set_xlim(self.axes_limits[0])
            ax.set_ylim(self.axes_limits[1])
            ax.set_zlim(self.axes_limits[2])
        
        ax.set_title("3D визуализация кластеров")
        ax.set_xlabel(self.df.columns[self.plot_features[0]])
        ax.set_ylabel(self.df.columns[self.plot_features[1]])
        ax.set_zlabel(self.df.columns[self.plot_features[2]])
        
        self.canvas.draw()
    
    def display_results(self, text):
        self.result_text.configure(state='normal')
        self.result_text.delete('1.0', END)
        self.result_text.insert(END, text)
        self.result_text.configure(state='disabled')

    def update_parameters(self):
        self.radius = self.radius_var.get()
        self.min_cluster_size = self.min_cluster_size_var.get()
        self.max_iter = self.max_iter_var.get()
        self.spa_features = self.spa_features_var.get()
        self.pop_size = self.pop_size_var.get()
        self.generations = self.generations_var.get()

    def run_clustering(self):
        self.update_parameters()
        start_time = time.time()
        
        labels = self.forel_cluster(self.df.values)
        self.compactness['all'] = self.compute_compactness(self.df.values, labels)
        n_clusters = len(np.unique(labels))
        
        self.display_results(
            f"FOREL (все признаки):\n"
            f"Кластеров: {n_clusters}\n"
            f"Компактность: {self.compactness['all']:.4f}\n"
            f"Время: {time.time()-start_time:.2f} сек"
        )
        self.update_plot(self.df.values, labels)
    
    def select_features(self):
        self.update_parameters()
        start_time = time.time()
        
        self.best_features = self.spa_feature_selection(
            self.df.values, 
            self.spa_features
        )

        self.display_results(
            f"Отбор признаков (СПА):\n"
            f"Выбраны: {self.df.columns[self.best_features].tolist()}\n"
            f"Время: {time.time()-start_time:.2f} сек"
        )
    
    def run_selected_clustering(self):
        if self.best_features is None:
            self.display_results("Сначала выполните отбор признаков (шаг 2)!")
            return
        
        self.update_parameters()
        start_time = time.time()
        
        X = self.df.values[:, self.best_features]
        labels = self.forel_cluster(X)
        self.compactness['selected'] = self.compute_compactness(X, labels)
        n_clusters = len(np.unique(labels))
        
        self.display_results(
            f"FOREL (выбранные признаки):\n"
            f"Кластеров: {n_clusters}\n"
            f"Компактность: {self.compactness['selected']:.4f}\n"
            f"Время: {time.time()-start_time:.2f} сек"
        )
        # Все равно используем исходные признаки для визуализации
        self.update_plot(self.df.values, labels)
    
    def anonymize_data(self):
        self.anonym_df = self.df.apply(lambda col: 
            (col // (col.std() / 3)) * (col.std() / 3)
        )
        self.display_results("Данные обезличены (дискретизация по σ/3)!")
    
    def run_anonym_clustering(self):
        if self.anonym_df is None:
            self.display_results("Сначала обезличьте данные (шаг 4)!")
            return
        
        self.update_parameters()
        start_time = time.time()
        
        labels = self.forel_cluster(self.anonym_df.values)
        self.compactness['anonym'] = self.compute_compactness(self.anonym_df.values, labels)
        n_clusters = len(np.unique(labels))
        
        self.display_results(
            f"FOREL (обезличенные данные):\n"
            f"Кластеров: {n_clusters}\n"
            f"Компактность: {self.compactness['anonym']:.4f}\n"
            f"Время: {time.time()-start_time:.2f} сек"
        )
        # Используем исходные признаки для визуализации
        self.update_plot(self.df.values, labels)
    
    def run_anonym_selected_clustering(self):
        if self.anonym_df is None:
            self.display_results("Сначала обезличьте данные (шаг 4)!")
            return
        if self.best_features is None:
            self.display_results("Сначала отберите признаки (шаг 2)!")
            return
        
        self.update_parameters()
        start_time = time.time()
        
        X = self.anonym_df.values[:, self.best_features]
        labels = self.forel_cluster(X)
        self.compactness['anonym_selected'] = self.compute_compactness(X, labels)
        n_clusters = len(np.unique(labels))
        
        self.display_results(
            f"FOREL (обезличенные + выбранные признаки):\n"
            f"Кластеров: {n_clusters}\n"
            f"Компактность: {self.compactness['anonym_selected']:.4f}\n"
            f"Время: {time.time()-start_time:.2f} сек"
        )
        # Все равно используем исходные признаки для визуализации
        self.update_plot(self.df.values, labels)

if __name__ == "__main__":
    app = ClusterLabApp()
    app.mainloop()
