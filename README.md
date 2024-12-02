# Website Fingerprinting

<p align="center">
  <a href="#setup">Setup</a> •
  <a href="#folder-structure">Folder Structure</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#results">Results</a> •
  <!-- <a href="#performance">Performance</a> •
  <a href="#insights">Insights</a> -->
</p>

## <a id="setup">Set Up</a>

### ☑️ Check if Conda is Installed

Before proceeding, verify whether **Conda** is already installed on your system.

1. **Open a terminal or command prompt**:

   - On **Windows**, use Command Prompt or Anaconda Prompt.
   - On **macOS/Linux**, open your terminal.

2. **Run the following command**:

   ```
   conda --version
   ```

   - If Conda is installed, you will see the version number(e.g., conda 24.1.2 for my system).
   - If Conda is not installed, proceed to install [Anaconda or miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

3. **Then you are good to go!**

### ☑️ Clone this Repository

To clone an existing repository: [FOLLOW STEPS 1-3](https://egene-chung.notion.site/Github-Repo-Clone-1204762b20cc800eb7ebfc06f639e3c1?pvs=4)

```bash
# Clone this repository
$ git clone https://github.com/egene-chung/website-fingerprinting.git

# Go into the repository
$ cd website-fingerprinting

# Create conda environment
$ conda create --name w-fp --file requirements.txt

# Activate conda environment
$ conda activate w-fp

# Run the model
$ cd src
$ python main.py
```

> Due to size constraints, the datasets have not been included in the pkl folder.
> You can, however, utilize the preprocessed CSV files generated by preprocess.py.
> If you prefer to create the CSV files yourself, please follow the folder structure outlined below:

- Upload mon_standard.pkl and unmon_standard10.pkl from the provided dataset into the data/pkl folder.
- Then, simply execute main.py! The script will automatically verify the existence of the required files and proceed accordingly.

## <a id="folder-structure">Folder Structure</a>

```text

├── 📁 src/
│   ├── 📁 data/
│   │   ├── 📁 csv/
│   │   │   ├── 📉 closedworld_data.csv
│   │   │   ├── 📉 openworld_binary_data.csv
│   │   │   └── 📉 openworld_multi_data.csv
│   │   ├── 📁 pkl/
│   │   │   ├── 📉 mon_statndard.pkl
│   │   │   └── 📉 unmon_standard10.pkl
│   │   └── 📄 preprocess.py
│   │
│   ├── 📁 experiments/
│   │   └── ...
│   │
│   ├── 📁 feature_selection/
│   │   ├── 📄 cw_multi_backward.py
│   │   ├── 📄 cw_multi_forward.py   ✅
│   │   ├── 📄 ow_binary_backward.py ✅
│   │   ├── 📄 ow_binary_forward.py
│   │   ├── 📄 ow_multi_backward.py  ✅
│   │   └── 📄 ow_multi_forward.py
│   ├── 📁 main_experiments/
│   │   ├── 📄 cw_random_forest.py
│   │   ├── 📉 cw_random_forest_results.csv
│   │   ├── 📄 ow_binary_random_forest.py
│   │   ├── 📉 ow_binary_random_forest_results.csv
│   │   ├── 📄 ow_multi_random_forest.py
│   │   └── 📉 ow_multi_random_forest_results.csv
│   │
│   ├── 📁 main_results/
│   │   ├── 📄 ow_binary_precision_recall_curve.png
│   │   ├── 📉 ow_binary_roc_curve.png
│   │   ├── 📄 results.log
│   │   └── 📉 sampling_results.log
│   │
│   ├── 📄 main.py
│   ├── 📄 sampling.py
│   └── 📄 evaluate.py
│
├── 📄 .gitignore
├── 📄 README.md
└── 📄 .requirements.txt
```

## <a id="experiments">Experimental Settings and Process Overview</a>

### 1. Iterative Hyperparameter Tuning with Stratified K-Fold

- Conduct hyperparameter tuning to maximize accuracy across folds:
  - Navigate to the `main_experiments/` folder.
  - Run all Python scripts in the folder.
  - Final results will be saved as `.txt` files in the same directory.
  - Identify the best parameters with the highest mean fold accuracy.

### 2. Feature Selection Using Forward Selection and Backward Elimination

- Select features to improve model performance:
  - Navigate to the `feature_selection/` folder.
  - Run all Python scripts in the folder. (The files with ✅ marks give best results)
  - Final results will be saved as `.txt` files in the same directory.
  - Use the parameters corresponding to the highest accuracy.

### 3. Applying Sampling Techniques in an Open-World Scenario

- Evaluate sampling methods to handle data imbalance:
  - Run the script `sampling.py` to apply sampling techniques.
  - Results previously saved can be found in `/main_results/sampling_results.log`

---

### Example Command Usage

```bash
# Run the model with best parameters and compare pre- and post-feature selection results.
# Previous results are stored in /main_results/results.log
$ python main.py

# Apply sampling techniques without feature selection
# Previous results are stored in /main_results/sampling_results.log
$ python sampling.py

```

## <a id="results">Results</a>

### Hyperparameter tuned model with, without Feature Selection

<img width="803" alt="image" src="https://github.com/user-attachments/assets/ba61fc51-3fcf-4f3d-b40e-99514881440d">

[Open World Binary Precision Recall Curve]

![image](https://github.com/user-attachments/assets/a8052bbf-6f6b-495c-8bf4-7b8b98c46b57)

[Open World Binary ROC Curve]

![image](https://github.com/user-attachments/assets/ee352115-ce8c-4d4a-b5a2-806ccaeb1cb9)

### Sampling applied without Feature Selection

<img width="805" alt="image" src="https://github.com/user-attachments/assets/defc9cd8-a9bc-45c6-8d38-47949c35b7d6">

<!-- ## <a id="problem-statement">3. Problem Statement</a>

## <a id="experimental-settings">4. Experimental Settings</a>

## <a id="model-selection">5. Model + Hyperparameter selection</a>

## <a id="experiments">6. Experiments</a>

## <a id="results">7. Results</a>

## <a id="performance">8. Performance</a>

## <a id="insights">9. Insights</a> -->

> If you encounter any issues starting the code, please contact (egenechung@gmail.com).
