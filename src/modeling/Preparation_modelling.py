import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# Suppress specific font warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
plt.rcParams['font.family'] = 'DejaVu Sans'

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    print(df.shape)
    df.drop_duplicates(inplace=True)
    print(df.shape)
    print(df.columns)
    columns_to_delete = ['Author', 'OriginalPaperDate', 'RetractionNature', 'RetractionDate']
    df.drop(columns=columns_to_delete, inplace=True)
    print(df.shape)
    return df

def count_occurrences(df, column):
    occurrences = {}
    for cell in df[column].dropna():
        entries = re.split(r'[;,]', cell)
        for entry in entries:
            entry = entry.strip()
            if entry:
                occurrences[entry] = occurrences.get(entry, 0) + 1
    return occurrences

def replace_with_most_frequent(df, column, occurrences):
    for i, cell in df[column].dropna().iteritems():
        entries = re.split(r'[;,]', cell)
        if len(entries) > 1:
            most_frequent_entry = max(entries, key=lambda x: occurrences.get(x.strip(), 0))
            df.loc[i, column] = most_frequent_entry.strip()

def preprocess_columns(df, columns_to_analyze):
    for column in columns_to_analyze:
        occurrences = count_occurrences(df, column)
        replace_with_most_frequent(df, column, occurrences)
    df['Reason'] = df['Reason'].str.replace('+', '', regex=False)
    return df

def perform_clustering(df, n_clusters=3):
    features = df[['TimeDifference_Days']]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(features)
    
    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data_scaled)
        inertia.append(kmeans.inertia_)

    # Save the Elbow Method plot to the figures directory
    figures_dir = os.path.normpath(os.path.join(base_dir, '../../results/figures'))
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    plt.figure(figsize=(10, 6))
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.savefig(os.path.join(figures_dir, 'elbow_method.png'))
    plt.show()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(data_scaled)
    return df

def filter_top_categories(df, column, top_n=10):
    top_categories = df[column].value_counts().nlargest(top_n).index
    df[column] = df[column].where(df[column].isin(top_categories), 'Other')
    return df

def analyze_and_plot(df):
    specific_categorical_features = ['Subject', 'Paywalled', 'Journal', 'Publisher', 'Country', 'ArticleType', 'Reason']
    
    figures_dir = os.path.normpath(os.path.join(base_dir, '../../results/figures'))
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    for feature in specific_categorical_features:
        print(f"\nDistribution of {feature} across Clusters:")
        print(df.groupby(['Cluster', feature]).size().unstack(fill_value=0))
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='Cluster', hue=feature)
        plt.title(f'Distribution of {feature} across Clusters')
        plt.savefig(os.path.join(figures_dir, f'distribution_{feature}_clusters.png'))
        plt.show()

def main():
    # Determine the base directory of the script
    global base_dir
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the file path for the input CSV file
    input_file_name = 'retraction_before_modelling.csv'
    input_file_path = os.path.normpath(os.path.join(base_dir, '../../data/processed', input_file_name))
    
    # Load and clean the data
    df = load_and_clean_data(input_file_path)
    print(df.shape)
    
    # Define the columns to analyze
    columns_to_analyze = ['Subject', 'Journal', 'Publisher', 'Country', 'ArticleType', 'Reason']
    
    # Preprocess the specified columns
    df = preprocess_columns(df, columns_to_analyze)
    print(df.shape)
    
    # Perform clustering
    df = perform_clustering(df, n_clusters=3)
    
    # Save the DataFrame after preprocessing and clustering
    output_file_name = 'preprocessed_data_with_clusters.csv'
    output_file_path = os.path.normpath(os.path.join(base_dir, '../../data/processed', output_file_name))
    df.to_csv(output_file_path, index=False)
    print(f"Preprocessed data with cluster labels saved to '{output_file_path}'")
    print(df.columns)
    
    # Filter top categories for the specified columns
    for feature in columns_to_analyze:
        df = filter_top_categories(df, feature)
    
    # Analyze and plot the data
    analyze_and_plot(df)
    print(df.shape)

if __name__ == "__main__":
    main()
