# Lab Projects
# Semester 3

## Lab 1: Synthetic Dataset Generator for Paid Polyclinic

### **Overview**
This project is a Python-based tool for generating synthetic datasets related to patient visits at a paid polyclinic. It creates realistic data for personal information, symptoms, doctor choices, analyses, and payment details, with customizable constraints.

### **Features**
- **Dataset Size**: Generates datasets with a minimum of 50,000 records.
- **Customizable Properties**:
  - Full Name: Based on a dictionary of Slavic names.
  - Passport Data: Includes Russian, Belarusian, and Kazakh passports.
  - Unique SNILS: Linked to personal information, supporting repeated visits.
  - Symptoms: A dictionary of over 5,000 symptoms.
  - Doctor Selection: A pool of at least 50 specialties.
  - Visit Dates: Within working hours, avoiding weekends.
  - Analyses: A selection of 250+ tests.
  - Payment Cards: Configurable likelihood for banks and payment systems.
- **Realistic Constraints**: Generates realistic timelines, costs in rubles, and recurring payment scenarios.

### **Setup**
1. Install Python.
2. Clone the GitHub repository.
3. Navigate to the project directory.

### **Usage**
1. Run the script:
   ```bash
   python main.py
   ```
2. Configure the settings for bank weights, payment systems, and dataset size.
3. The dataset will be saved as `dataset.csv` in the project directory.

### **Dataset Specifications**
- **General**:
  - Minimum 50,000 rows.
  - Slavic-origin names only.
  - Unique SNILS linked to patient identity (repeated for multiple visits).
- **Medical Constraints**:
  - Symptoms: At least 5,000 unique entries; up to 10 per patient.
  - Doctors: At least 50 specialties.
  - Analyses: Minimum of 250 types; combinations up to 5 per patient.
  - Visit Dates: Within working hours and days; subsequent visits possible 24+ hours after analysis receipt.
  - Analysis Receipt: Within 24–72 hours after collection, during working hours.
- **Payment Constraints**:
  - Transactions in rubles only.
  - Card details support configurable probabilities for banks and payment systems.
  - Maximum of 5 transactions per card.

---

## Lab 2: Data Anonymization Tool

### **Overview**
This tool provides functionality for anonymizing datasets and assessing k-anonymity, ensuring data privacy while maintaining utility for analysis. It is designed as a second-stage process for anonymizing the synthetic dataset generated in the first lab.

### **Features**
1. **Data Loading**:
   - Read input datasets (output from Lab 1).
2. **Quasi-Identifiers**:
   - Select quasi-identifiers for anonymization via an interactive interface.
3. **Anonymization Methods**:
   - Techniques include:
     - **Local Generalization**: Group similar values.
     - **Aggregation**: Summarize data into broader categories.
     - **Perturbation**: Introduce noise or randomization.
     - **Micro-Aggregation**: Replace values with group averages.
     - **Swapping**: Randomly reorder values.
     - **Pseudonymization**: Replace data with aliases.
     - **Masking**: Partially obscure sensitive attributes.
     - **Local Suppression**: Remove specific values.
     - **Attribute Deletion**: Exclude fields entirely.
     - **Decomposition**: Break datasets into smaller, less-identifiable subsets.
4. **k-Anonymity Calculation**:
   - Compute k-anonymity before and after anonymization.
   - Identify "bad" k-anonymity values and recommend improvements.
5. **Utility Evaluation**:
   - Compare anonymized and original datasets to assess utility loss.

### **Setup**
1. Install Python and required libraries:
   ```bash
   pip install pandas
   ```
2. Clone the repository from GitHub.
3. Place the input dataset (`dataset.csv`) in the `2 Lab/` directory.

### **Usage**
1. **Run the script**:
   ```bash
   python main.py
   ```
2. **Follow Prompts**:
   - Indicate which fields to anonymize.
3. **Select Quasi-Identifiers**:
   - Choose from fields like `Full Name`, `Passport Data`, `SNILS`, etc.
4. **Anonymization and k-Anonymity**:
   - Anonymize the dataset and compute k-anonymity.

### **Output**
- Anonymized dataset saved as `2 Lab/anon_dataset.csv`.
- Details of "bad" k-anonymity values in `2 Lab/bad_k_values.csv`.
- "Good" k-anonymity values stored in `2 Lab/good_k_values.csv`.

### **Evaluation**
- Identifies acceptable k-anonymity thresholds:
  - Up to 51,000 rows: \( k \geq 10 \)
  - Up to 105,000 rows: \( k \geq 7 \)
  - Up to 260,000 rows: \( k \geq 5 \)
- Reports percentage and count of "bad" k-anonymity rows.
- Displays unique rows where \( k = 1 \) for deeper insights.

---

## Lab 3: Hash Function Research Project

### **Overview**
This project explores the decryption of datasets encrypted using hash functions with an input modifier (salt). The objective is to study the behavior and efficiency of decryption under varying conditions and with different hash functions.

### **Features**
- **Phone Number Deanonymization**: Identifies salts used during encryption and reverses the process.
- **Hashing Algorithms**: Encrypts data using multiple hash functions and salt types.
- **Performance Analysis**: Measures decryption speed under various scenarios.

### **Goals**
1. Analyze encryption methods for phone numbers.
2. Implement a program to deanonymize datasets using salts.
3. Test the program with provided datasets and additional hash functions (MD5, SHA-1, Blake2b).
4. Investigate factors affecting decryption speed, including salt type, salt length, and hash algorithms.
5. Determine the minimum dataset size required for complete decryption.

### **Setup**
1. Install Python and required libraries.
2. Clone the repository from GitHub.
3. Run the scripts for analysis and encryption:
   ```bash
   python phones_deanonymization.py
   python hashing.py
   ```
4. Use Hashcat to test decryption efficiency and validate results.

### **Usage**
1. Load the dataset to analyze matching hashes and salts.
2. Deanonymize the data to recover original phone numbers.
3. Generate new encrypted datasets using different algorithms and salt sizes.

---

## Lab 4: Genetic Algorithm Exploration: Encoding Methods

### **Overview**
This project explores two primary chromosome genotype encoding methods in genetic algorithms, evaluating their effectiveness in solving optimization problems.

### **Features**
- **Function to Minimize**:
  - f(x_1, x_2) = x_1^3 + x_2^2 - 3x_1 - 2x_2 + 2, aiming to find the minimum value.
- **Genetic Algorithm Implementation**: Includes mutation, crossover, selection, and fitness evaluation.
- **Encoding Modes**: Supports integer and real-number-based genotype encoding.
- **Algorithm Modes**: Switch between standard and modified genetic algorithm implementations.
- **GUI Integration**: A user-friendly interface for adjusting parameters, running the algorithm, and visualizing results.


### **Setup**
1. Install Python and required libraries:
2. Clone the repository from GitHub.
3. Run the script:
   ```bash
   python main.py
   ```
4. Adjust parameters in the GUI and click "Calculate" to execute the algorithm.
5. Visualize results and analyze the performance of different encoding methods.

---

## Lab 5: Swarm Intelligence Algorithm for Global Optimization

### **Overview**
This project implements a Swarm Intelligence algorithm (Particle Swarm Optimization, PSO) to solve global optimization problems and provides an interface to visualize the particle movements and optimization process.

### **Features**
- **Function to Minimize**:
  - f(x_1, x_2) = x_1^3 + x_2^2 - 3x_1 - 2x_2 + 2, aiming to find the minimum value.
- **Particle Swarm Optimization (PSO)**:
  - The algorithm searches for the minimum of a given test function by simulating a swarm of particles.
  - Particles adjust their position and velocity based on their personal best and the swarm’s global best.
- **Modified and Standard PSO Versions**: Includes both standard and modified PSO for comparison.
- **Visualization**: Real-time display of particle positions and optimization progress.

### **User Interface**
- Configurable parameters: inertia, personal best coefficient, global best coefficient, number of particles, and iterations.
- Real-time visualization of particle movements and convergence to the global optimum.

### **Setup**
1. Install dependencies:
   ```bash
   pip install numpy matplotlib
   ```
2. Run the script:
   ```bash
   python main.py
   ```
3. Use the GUI to configure parameters and start the algorithm.

### **Output**
- Visual representation of particle movements.
- Best solution and fitness value displayed at the end of the run.

---

# Semester 4

## Lab 1: Traveling Salesman Problem - Nearest Neighbor Algorithm  

### **Overview**  
This project implements a graphical solution to the Traveling Salesman Problem (TSP) using the **Nearest Neighbor Algorithm**. The application provides an interactive interface for constructing weighted directed graphs, finding the shortest Hamiltonian cycle, and visualizing the optimal path.  

### **Features**  
- **Interactive Graph Editor**:  
  - Add nodes by clicking on the canvas.  
  - Create weighted edges by selecting two nodes (automatically calculates Euclidean distance).  
- **TSP Solver**:  
  - Finds the shortest cycle using the greedy nearest-neighbor approach.  
  - Supports optimization by testing all possible starting nodes (optional).  
- **Visualization**:  
  - Displays the original graph and the computed solution side-by-side.  
  - Highlights the optimal path with directional arrows.  
- **Data Management**:  
  - Save/load graphs in JSON format.  
  - Undo/clear functionality for graph editing.  

### **Technical Implementation**  
- **Algorithm**:  
  - Starts from a node, iteratively visits the nearest unvisited neighbor.  
  - Optional optimization: tests every node as a starting point for better solutions.  
- **Constraints**:  
  - Handles directed edges (non-symmetric distances).  
  - Requires a complete cycle (returns to the starting node if possible).  
- **Performance**:  
  - Time complexity: *O(n²)* per starting node (or *O(n)* for a fixed start).  

### **Setup**  
1. Install Python and required libraries:
   ```bash
   pip install numpy
   ```
2. Clone the repository from GitHub.
3. Use the input graph (`graph.json`) in the `2 semester/1 Lab/` directory.

### **Usage**  
1. **Build the Graph**:  
   - Left-click to add nodes.  
   - Select two nodes to create an edge (weight = Euclidean distance).  
2. **Solve TSP**:  
   - Click *"Рассчитать"* to compute the shortest cycle.  
   - Enable *"Использовать модификацию"* to test all starting nodes.  
3. **Save/Load**:  
   - Use *"Загрузить граф"* to import a pre-built graph (JSON format).  

---  

## Lab 2: Traveling Salesman Problem - Simulated Annealing  

### **Overview**  
This project implements a **Simulated Annealing** algorithm with an optional *"Fast Annealing"* modification to solve the Traveling Salesman Problem (TSP). The application provides comparative analysis with the Nearest Neighbor method from Lab 1, evaluating performance on graphs of varying sizes.  

### **Key Features**  
- **Two Cooling Strategies**:  
  - Standard exponential cooling (`T = T0 * α^k`)  
  - Fast annealing (`T = T0 / (1 + k)`)  
- **Interactive Graph Interface**:  
  - Visual graph construction with weighted edges  
  - Hamiltonian cycle validation  
- **Performance Metrics**:  
  - Execution time tracking  
  - Path length optimization  
- **Comparative Analysis**:  
  - Benchmarking against Nearest Neighbor results  

### **Technical Implementation**  
- **Algorithm Core**:  
  ```python
  if fast_annealing:
      T = T0 / (1 + k)
  else: 
      T = T0 * (α ** k)
  ```  
- **Neighbor Selection**:  
  - 2-opt swaps for local search  
  - Boltzmann probability acceptance: `exp(-ΔE/T)`  
- **Distance Matrix**: Handles asymmetric edge weights (directed graphs)  

### **Setup**  
1. Install Python and required libraries:
   ```bash
   pip install numpy
   ```
2. Clone the repository from GitHub.
3. Use the input graph (`graph.json`) in the `2 semester/1 Lab/` directory.

### **Usage**  
1. **Parameters**:  
   - Initial temperature (`T0=5000`)  
   - Iterations (`max_iter=10000`)  
   - Cooling rate (`α=0.5`)  
2. **Workflow**:  
   ```plaintext
   1. Build graph → 2. Set parameters → 3. Run → 4. Compare with Lab 1
   ```  
3. **Hotkeys**:  
   - Left-click: Add node  
   - Select two nodes: Create edge  

---  

## Lab 3: Traveling Salesman Problem - Ant Colony Optimization  

### **Overview**  
This project implements an **Ant Colony Optimization (ACO)** algorithm with a *"Wandering Colony"* modification to solve the Traveling Salesman Problem. The solution demonstrates swarm intelligence principles through pheromone-based path optimization, with comparative analysis against previous methods (Nearest Neighbor and Simulated Annealing).

### **Key Features**  
- **Dual Algorithm Modes**:  
  - Standard ACO with random starting cities  
  - *Wandering Colony* modification (all ants start from same city)  
- **Interactive Parameter Control**:  
  - Adjustable pheromone influence (α)  
  - Distance heuristic weight (β)  
  - Evaporation rate customization  
- **Visual Analytics**:  
  - Real-time solution visualization  
  - Side-by-side graph comparison  

### **Technical Implementation**  
```python
# Core probability calculation
probability = (τ^α) * (η^β) / Σ(τ^α * η^β)
where:
τ = pheromone level  
η = 1/distance (heuristic)  
α/β = influence parameters
```

- **Pheromone Update**:  
  ```python
  pheromone *= (1 - evaporation)  
  pheromone[best_path] += Q/path_length
  ```  
- **Path Construction**:  
  - Roulette wheel selection for next node  
  - 2-opt local search integration  

### **Setup**  
1. Install Python and required libraries:
   ```bash
   pip install numpy
   ```
2. Clone the repository from GitHub.
3. Use the input graph (`graph.json`) in the `2 semester/1 Lab/` directory. 

### **Usage**  
1. **Graph Setup**:  
   - Left-click to add nodes  
   - Select two nodes to create weighted edges  
2. **Parameter Configuration**:  
   ```python
   Recommended settings:
   - Ants: 10-20
   - Iterations: 50-200  
   - α=1, β=2-5  
   - Evaporation: 0.3-0.7
   ```  
3. **Execution**:  
   - Enable *"Wandering Colony"* for focused search  
   - Compare results with Lab 1/2 datasets  

---

## Lab 4: Data Recovery - Missing Value Imputation  

### **Overview**  
This project implements three methods for missing data imputation in medical datasets: **Hot-Deck**, **Mean Value**, and **Linear Regression**. The application evaluates imputation quality through statistical error metrics and comparative analysis of restored distributions.

### **Key Features**  
- **Multi-Method Imputation**:  
  - **Hot-Deck**: Random sampling from existing values  
  - **Mean Value**: Column-average replacement  
  - **Linear Regression**: Predictive modeling for numeric fields  
- **Dataset Handling**:  
  - Supports small (~10K), medium (~75K), and large (~250K) datasets  
  - Controlled missing value injection (3-30% gaps)  
- **Error Analysis**:  
  - Categorical vs numeric error differentiation  
  - Mean absolute percentage error (MAPE) calculation  

### **Imputation Methods**  
| Method            | Best For                | 
|-------------------|-------------------------|
| Hot-Deck          | Categorical data        |
| Mean Value        | Normally-distributed    |
| Linear Regression | Correlated numeric      |

### **Technical Implementation**  
#### **Core Algorithms**  
```python
# Hot-Deck Imputation
df_filled.loc[missing, col] = np.random.choice(non_missing, size=missing.sum())

# Linear Regression
model = LinearRegression(positive=True)
model.fit(X_train, y_train)  
predicted = model.predict(X_test)
```

#### **Data Flow**  
1. **Missing Value Generation**:  
   ```python
   indices_to_drop = np.random.choice(df.index, size=n, replace=False)
   df_with_nans.loc[indices_to_drop, column] = np.nan
   ```  
2. **Regression Features**:  
   - One-hot encoded symptoms/doctor specialties  
   - Numeric analysis costs as target  

### **Setup**  
1. Install Python and required libraries:
   ```bash
   pip install numpy pandas scikit-learn
   ```
2. Clone the repository from GitHub.
3. Use the input dataset (`dataset.csv`) in the `1 semester/1 Lab/` directory.

### **Usage**  
1. Load CSV file  
2. Set deletion percentage (e.g., 20%)  
3. Output: `dataset_nan.csv`  

---  

## Lab 5: Data Clustering with FOREL Algorithm  

### **Overview**  
This project implements a comprehensive clustering pipeline using the **FOREL algorithm** with **Euclidean distance metric**, featuring **SPA feature selection** and **cluster compactness evaluation**. The system compares clustering quality across original, anonymized, and feature-selected datasets.

### **Key Features**  
- **Multi-Stage Clustering Pipeline**:  
  - Original data clustering  
  - Feature-selected clustering (SPA algorithm)  
  - Anonymized data clustering  
- **Quality Metrics**:  
  - Cluster compactness measurement  
  - Comparative analysis across different data treatments  
- **Visual Analytics**:  
  - 3D visualization of clusters  
  - Fixed-axis comparison for consistent evaluation  

### **Technical Implementation**  

#### **Core Algorithms**  
```python
# FOREL Clustering
def forel_cluster(X):
    while unassigned_points:
        center = random_unassigned_point()
        while True:
            in_radius = points_within(center, radius)
            new_center = centroid(in_radius)
            if distance(center, new_center) < epsilon:
                break
            center = new_center
        assign_to_cluster(in_radius)
```

#### **SPA Feature Selection**  
```python
# Evolutionary feature selection
population = [random_features() for _ in range(pop_size)]
for generation in generations:
    evaluate_fitness(population)  # Using cluster compactness
    new_population = [mutate(best_features) for _ in range(pop_size)]
```

### **Setup**  
1. Install Python and required libraries:
   ```bash
   pip install numpy pandas scikit-learn matplotlib scipy
   ```
2. Clone the repository from GitHub.

### **Usage**  

#### **1. Clustering Workflow**  
1. **Initial Clustering**:  
   - Run FOREL on all features  
   - Visualize in 3D space  
2. **Feature Selection**:  
   - Execute SPA algorithm  
   - Select top N informative features  
3. **Selected Feature Clustering**:  
   - Re-run FOREL on reduced feature set  
4. **Data Anonymization**:  
   - Apply σ/3 discretization  
5. **Anonymized Data Clustering**:  
   - Compare results with original  

#### **2. Parameter Configuration**  
| Parameter            | Default | Description                     |  
|----------------------|---------|---------------------------------|  
| Cluster radius       | 2.5     | FOREL neighborhood size         |  
| Min cluster size     | 10      | Minimum points per cluster      |  
| SPA features         | 5       | Number of features to select    |  
| SPA generations      | 50      | Evolutionary iterations         |  

---  
