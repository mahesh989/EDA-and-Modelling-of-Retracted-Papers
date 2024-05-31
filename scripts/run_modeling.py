import os
import sys

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.modeling.Preparation_modelling import main as prepare_data
from src.modeling.predictive_modelling_approach_2 import main as approach_2
from src.modeling.predictive_modeling_approach_3 import main as approach_3
from src.modeling.confusion_matrix_random_forest import main as generate_confusion_matrix

def main():
    # Step 1: Preparation
    print("Running data preparation...")
    prepare_data()

    # Step 2: Predictive Modeling Approach 1
    print("Running predictive modeling approach 1...")
    approach_2()

    # Step 3: Predictive Modeling Approach 2
    print("Running predictive modeling approach 2...")
    approach_3()

    # Step 4: Generate Confusion Matrix
    print("Generating confusion matrix...")
    generate_confusion_matrix()

if __name__ == '__main__':
    main()
