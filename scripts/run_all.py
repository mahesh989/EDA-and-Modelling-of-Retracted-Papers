import os
import subprocess

def run_script(script_path):
    try:
        print(f"Running script: {script_path}")
        result = subprocess.run(['python', script_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running script: {script_path}")
        print(f"Standard Output:\n{e.stdout}")
        print(f"Standard Error:\n{e.stderr}")
        raise

def main():
    # Determine the base directory of the script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the paths to the scripts
    scripts = [
        os.path.normpath(os.path.join(base_dir, '../src/eda/EDA_retraction.py')),
        os.path.normpath(os.path.join(base_dir, '../src/modeling/Preparation_modelling.py')),
        os.path.normpath(os.path.join(base_dir, '../src/modeling/predictive_modelling_approach_2.py')),
        os.path.normpath(os.path.join(base_dir, '../src/modeling/predictive_modeling_approach_3.py')),
        os.path.normpath(os.path.join(base_dir, '../src/modeling/confusion_matrix_random_forest.py'))
    ]
    
    # Run each script
    for script in scripts:
        run_script(script)

    print("All tasks completed successfully.")

if (__name__) == "__main__":
    main()


