import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import matplotlib.pyplot as plt

def preprocess_data(df):
    # Convert categorical columns to category type
    cat_cols = ['Subject', 'Journal', 'Publisher', 'Country', 'ArticleType', 'Reason', 'Paywalled']
    df[cat_cols] = df[cat_cols].astype('category')
    return df

def prepare_data(df):
    X = df.drop(columns=['TimeDifference_Days'])  # Features
    y = df['TimeDifference_Days']  # Target
    return X, y

def train_evaluate_model(model, X_train, y_train, X_test, y_test):
    # Train model
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    
    return mse, r2, mae, evs, y_test, y_pred

def main():
    # Determine the base directory of the script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the file path for the input CSV file
    input_file_name = 'preprocessed_data_with_clusters.csv'
    input_file_path = os.path.normpath(os.path.join(base_dir, '../../data/processed', input_file_name))
    
    # Read the data
    df = pd.read_csv(input_file_path)
    
    # Drop the "Cluster" column
    df.drop(columns=['Cluster'], inplace=True)
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Prepare data
    X, y = prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define categorical and numerical features
    categorical_features = ['Subject', 'Journal', 'Publisher', 'Country', 'ArticleType', 'Reason', 'Paywalled']
    numerical_features = ['CitationCount', 'Retraction_to_Citation_Ratio', 'Citation_to_Retraction_Ratio']
    
    # Define preprocessing steps
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)],
        remainder='passthrough'
    )
    
    # Models
    models = {
        'Linear Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ]),
        'SVR': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', SVR())
        ]),
        'Random Forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor())
        ])
    }
    
    # Create directory for figures if it does not exist
    figures_dir = os.path.normpath(os.path.join(base_dir, '../../results/figures'))
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    for name, model in models.items():
        print(f"\n{name} Model:")
        mse, r2, mae, evs, y_test, y_pred = train_evaluate_model(model, X_train, y_train, X_test, y_test)
        print("Mean Squared Error:", mse)
        print("R^2 Score:", r2)
        print("Mean Absolute Error:", mae)
        print("Explained Variance Score:", evs)
        
        # Plotting the results
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
        plt.xlabel('Measured')
        plt.ylabel('Predicted')
        plt.title(f'{name} Model: Measured vs Predicted')
        plt.savefig(os.path.join(figures_dir, f'{name}_measured_vs_predicted.png'))
        plt.show()

if __name__ == "__main__":
    main()
