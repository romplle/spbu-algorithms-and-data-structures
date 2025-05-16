import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

class DataRecoveryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Восстановление данных")
        self.root.geometry("900x400")
        
        self.source_file_path = ""
        self.restore_file_path = ""
        self.input_file_path = ""
        self.output_file_path = ""
        
        self.create_widgets()
    
    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка "Удаление данных"
        self.delete_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.delete_tab, text="Удаление данных")
        self.create_delete_tab()
        
        # Вкладка "Восстановление данных"
        self.restore_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.restore_tab, text="Восстановление данных")
        self.create_restore_tab()
        
        # Вкладка "Расчёт ошибки"
        self.error_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.error_tab, text="Расчёт ошибки")
        self.create_error_tab()
    
    def create_delete_tab(self):
        frame = ttk.Frame(self.delete_tab, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Исходный файл
        ttk.Label(frame, text="Исходный файл:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.source_file_entry = ttk.Entry(frame, state='readonly')
        self.source_file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        ttk.Button(frame, text="Выбрать", command=self.select_source_file).grid(row=0, column=2, sticky=tk.W, pady=5)

        # Удаление данных
        ttk.Label(frame, text="Удалить (%):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.percent_entry = ttk.Entry(frame)
        self.percent_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        ttk.Button(frame, text="Удалить данные", command=self.delete_data).grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

    def create_restore_tab(self):
        frame = ttk.Frame(self.restore_tab, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Файл для восстановления
        ttk.Label(frame, text="Файл для восстановления:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.restore_file_entry = ttk.Entry(frame, state='readonly')
        self.restore_file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        ttk.Button(frame, text="Выбрать", command=self.select_restore_file).grid(row=0, column=2, sticky=tk.W, pady=5)

        # Метод восстановления
        ttk.Label(frame, text="Метод восстановления:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.method_var = tk.StringVar()
        method_combobox = ttk.Combobox(frame, textvariable=self.method_var, state='readonly')
        method_combobox['values'] = ('Hot-deck', 'Mean', 'Linear_regression')
        method_combobox.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        method_combobox.current(0)
        ttk.Button(frame, text="Восстановить данные", command=self.restore_data).grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
    
    def create_error_tab(self):
        main_frame = ttk.Frame(self.error_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Левая панель
        control_frame = ttk.Frame(main_frame, width=500, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Правая панель
        output_frame = ttk.Frame(main_frame, padding=10)
        output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Элементы управления
        ttk.Label(control_frame, text="Исходный файл:").pack(anchor=tk.W, pady=5)
        self.input_file_name_entry = ttk.Entry(control_frame, state='readonly', width=50)
        self.input_file_name_entry.pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Выбрать", command=self.select_input_file).pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="Восстановленный файл:").pack(anchor=tk.W, pady=5)
        self.output_file_name_entry = ttk.Entry(control_frame, state='readonly', width=50)
        self.output_file_name_entry.pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Выбрать", command=self.select_output_file).pack(fill=tk.X, pady=5)

        ttk.Button(control_frame, text="Рассчитать ошибку", command=self.calculate_error).pack(fill=tk.X, pady=10)
        
        # Область вывода
        self.output_text = tk.Text(output_frame, wrap=tk.WORD, height=20, width=40)
        scrollbar = ttk.Scrollbar(output_frame, command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def select_source_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.source_file_path = file_path
            self.source_file_entry.configure(state='normal')
            self.source_file_entry.delete(0, tk.END)
            self.source_file_entry.insert(0, os.path.basename(file_path))
            self.source_file_entry.configure(state='readonly')

    def select_restore_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.restore_file_path = file_path
            self.restore_file_entry.configure(state='normal')
            self.restore_file_entry.delete(0, tk.END)
            self.restore_file_entry.insert(0, os.path.basename(file_path))
            self.restore_file_entry.configure(state='readonly')

    def select_input_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.input_file_path = file_path
            self.input_file_name_entry.configure(state='normal')
            self.input_file_name_entry.delete(0, tk.END)
            self.input_file_name_entry.insert(0, os.path.basename(file_path))
            self.input_file_name_entry.configure(state='readonly')

    def select_output_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.output_file_path = file_path
            self.output_file_name_entry.configure(state='normal')
            self.output_file_name_entry.delete(0, tk.END)
            self.output_file_name_entry.insert(0, os.path.basename(file_path))
            self.output_file_name_entry.configure(state='readonly')

    def delete_data(self):
        if not self.source_file_path:
            messagebox.showerror("Ошибка", "Исходный файл не выбран")
            return
        
        try:
            pct = float(self.percent_entry.get())
            if not 0 <= pct <= 100:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Введите число от 0 до 100")
            return
        
        df = pd.read_csv(self.source_file_path)
        
        df_with_nans = df.copy()
        
        for column in df.columns:
            n = max(1, int(len(df) * pct / 100))
            indices_to_drop = np.random.choice(df.index, size=n, replace=False)
            df_with_nans.loc[indices_to_drop, column] = np.nan
        
        out_path = os.path.splitext(self.source_file_path)[0] + "_nan.csv"
        df_with_nans.to_csv(out_path, index=False)
        
        removed = df_with_nans.isna().sum().sum()
        total = df.size
        
        messagebox.showinfo(
            "Готово",
            f"Удалено {removed} из {total} ячеек ({removed/total*100:.2f}%)\n"
            f"Файл сохранен: {out_path}"
        )

    def hot_deck_imputation(self, df):
        df_filled = df.copy()

        for col in df.columns:
            missing = df_filled[col].isna()
            if missing.any():
                non_missing = df_filled[col].dropna()
                if len(non_missing) > 0:
                    df_filled.loc[missing, col] = np.random.choice(non_missing, size=missing.sum())
        return df_filled

    def mean_imputation(self, df):
        df_filled = df.copy()
        numeric_cols = [col for col in df.select_dtypes(include=np.number).columns]
        
        for col in numeric_cols:
            df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
        return df_filled

    def regression_imputation(self, df):
        df_filled = df.copy()
        
        features = ['Выбор врача', 'Симптомы', 'Анализы']
        target = 'Стоимость анализов, руб'
        
        temp_df = df_filled.copy()
        
        for feature in features:
            dummies = temp_df[feature].str.get_dummies(sep=', ')
            dummies.columns = [f"{feature}_{col}" for col in dummies.columns]
            temp_df = pd.concat([temp_df, dummies], axis=1)
        
        X = temp_df.drop(columns=features + [target] + [
            'ФИО', 'Паспортные данные', 'СНИЛС', 
            'Дата посещения врача', 'Дата получения анализов',
            'Номер карты', 'Банк', 'Платёжная система'
        ], errors='ignore')
        
        y = temp_df[target]
        
        train_mask = y.notna()
        test_mask = y.isna()
        
        if not train_mask.any() or not test_mask.any():
            return df_filled
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        
        try:
            model = LinearRegression(positive=True)
            model.fit(X_train, y_train)
            
            predicted = model.predict(X_test)
            df_filled.loc[test_mask, target] = predicted.round(2)
        except Exception as e:
            print(f"Ошибка при обучении модели: {e}")
        
        return df_filled
                    
    def recover_data(self, df, method):
        if method == 'Hot-deck':
            return self.hot_deck_imputation(df)
        elif method == 'Mean':
            return self.mean_imputation(df)
        elif method == 'Linear_regression':
            return self.regression_imputation(df)
        return df.copy()

    def restore_data(self):
        if not self.restore_file_path:
            messagebox.showerror("Ошибка", "Файл для восстановления не выбран")
            return

        df = pd.read_csv(self.restore_file_path)
        df_recovered = self.recover_data(df, self.method_var.get())
        
        out_path = os.path.splitext(self.restore_file_path)[0] + f"_{self.method_var.get()}.csv"
        df_recovered.to_csv(out_path, index=False)
        messagebox.showinfo("Готово", f"Файл сохранен: {out_path}")

    def calculate_error(self):
        df_original = pd.read_csv(self.input_file_path)
        df_restored = pd.read_csv(self.output_file_path)
        
        categorical_cols = df_original.select_dtypes(include=['object', 'category']).columns
        numeric_cols = df_original.select_dtypes(include=['int64', 'float64']).columns

        numeric_errors = []
        categorical_errors = []
        self.output_text.delete(1.0, tk.END)
        
        for col in categorical_cols:
            error_count = 0
            total_count = 0
            original_values = df_original[col].values
            restored_values = df_restored[col].values
            
            for orig, rest in zip(original_values, restored_values):
                if pd.isna(orig):
                    continue
                    
                total_count += 1
                
                if pd.isna(rest) or orig != rest:
                    error_count += 1
            
            if total_count > 0:
                error_percent = (error_count / total_count) * 100
                categorical_errors.append(error_percent)
                self.output_text.insert(tk.END, f"{col} (кат.): {error_percent:.2f}%\n")
            else:
                self.output_text.insert(tk.END, f"{col} (кат.): Нет данных для расчёта\n")

        for col in numeric_cols:
            numeric_error = 0
            numeric_count = 0
            original_values = df_original[col].values
            restored_values = df_restored[col].values
            
            for orig, rest in zip(original_values, restored_values):
                numeric_count += 1
                numeric_error += abs(orig - rest) / abs(orig)
            
            if numeric_count > 0:
                error_percent = (numeric_error / numeric_count) * 100
                numeric_errors.append(error_percent)
                self.output_text.insert(tk.END, f"{col} (числ.): {error_percent:.2f}%\n")
            else:
                self.output_text.insert(tk.END, f"{col} (числ.): Нет данных для Расчёта\n")
            
        if categorical_errors:
            avg_cat = sum(categorical_errors)/len(categorical_errors)
            self.output_text.insert(tk.END, f"Средняя ошибка (кат.): {avg_cat:.2f}%\n")

        if numeric_errors and categorical_errors:
            overall = (sum(numeric_errors) + sum(categorical_errors)) / (len(numeric_errors) + len(categorical_errors))
            self.output_text.insert(tk.END, f"Общая ошибка: {overall:.2f}%\n")
        elif categorical_errors:
            self.output_text.insert(tk.END, f"Общая ошибка: {avg_cat:.2f}% (только кат.)\n")
        else:
            self.output_text.insert(tk.END, "Нет данных для расчёта ошибок\n")

root = tk.Tk()
app = DataRecoveryApp(root)
root.mainloop()
