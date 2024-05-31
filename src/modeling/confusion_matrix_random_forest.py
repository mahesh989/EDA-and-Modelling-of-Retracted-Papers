
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_classification_report(df, output_dir):
    # Define categorical and numerical features
    cat_cols = ['Subject', 'Journal', 'Publisher', 'Country', 'ArticleType', 'Reason', 'Paywalled']
    numerical_features = ['CitationCount', 'Retraction_to_Citation_Ratio', 'Citation_to_Retraction_Ratio', 'TimeDifference_Days']

    # Split the dataset into features (X) and target (y)
    X = df.drop(columns=['Cluster'])  # Features
    y = df['Cluster']  # Target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing steps
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, cat_cols),
            ('num', numerical_transformer, numerical_features)],
        remainder='passthrough'
    )

    # Train a Random Forest classifier
    classifier = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])

    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Display confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', annot_kws={"size": 12})
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))  # Save the plot
    plt.show()

    # Additional metrics (classification report)
    classification_rep = classification_report(y_test, y_pred)
    print(classification_rep)

    # Save classification report to a text file
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(classification_rep)

def main():
    # Determine the base directory of the script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the file path for the input CSV file
    input_file_name = 'preprocessed_data_with_clusters.csv'
    input_file_path = os.path.normpath(os.path.join(base_dir, '../../data/processed', input_file_name))
    
    # Read the data
    df = pd.read_csv(input_file_path)

    # Output directory
    output_dir = os.path.normpath(os.path.join(base_dir, '../../classification_results'))

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate classification report
    generate_classification_report(df, output_dir)

if __name__ == "__main__":
    main()
