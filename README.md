# Lab Projects: Semester 3

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
3. Place the input dataset (`dataset.csv`) in the `1 Lab/` directory.

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