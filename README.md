
# EDA and Modelling of Retracted Papers

## Project Overview
This project performs exploratory data analysis (EDA) and predictive modeling on retracted papers. The aim is to understand the characteristics of retracted papers and to develop models to predict retractions.

## Directory Structure

```
your-github-repo/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── eda/
│   │   └── EDA_retraction.py
│   ├── modeling/
│   │   ├── Preparation_modelling.py
│   │   ├── predictive_modelling_approach_2.py
│   │   ├── predictive_modeling_approach_3.py
│   │   └── confusion_matrix_random_forest.py
│
├── results/
│   ├── figures/
│   └── reports/
│
└── scripts/
    ├── run_modeling.py
    └── run_all.py
```

## Setup Instructions

### Prerequisites
- Python 3.6 or higher
- Git (optional, for cloning the repository)

### Steps to Run the Project

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Create a Virtual Environment** (Recommended):
   

3. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the python file**:
   For example
   ```sh
   python scripts/run_all.py
   ```

## Explanation of Key Scripts

### src/eda/EDA_retraction.py
This script performs exploratory data analysis on the retracted papers dataset. It generates visualizations and descriptive statistics to understand the characteristics of the data. Before this data cleaning has been done. 

### src/modeling/Preparation_modelling.py
This script prepares the data for modeling by preprocessing and transforming the dataset. It ensures the data is in the correct format for the predictive models.

### src/modeling/predictive_modelling_approach_1.py
check https://github.com/bibekdhakal/research-retraction

### src/modeling/predictive_modelling_approach_2.py
This script implements the second approach for predictive modeling. It trains and evaluates a machine learning model to predict retractions.

### src/modeling/predictive_modeling_approach_3.py
This script implements the third approach for predictive modeling. It trains and evaluates another machine learning model to predict retractions.
Applying clustering techniques (e.g., K-Means) to group similar data points, thereby capturing underlying patterns in the data.

### src/modeling/confusion_matrix_random_forest.py
This script generates a confusion matrix for the Random Forest model. It evaluates the performance of the model and visualizes the results.

### scripts/run_all.py
This script orchestrates the execution of all the key scripts in the correct order. It ensures that the entire workflow from data preparation to model evaluation is completed.

## Results
The results of the analysis and modeling are saved in the `results` directory. This includes figures. Based on this a report has been made using Texmaker.

## Contact
For any questions or issues, please contact Mahesh Tiwari at maheshtwari99@gmail.com
