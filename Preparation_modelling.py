import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

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

    plt.figure(figsize=(10, 6))
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal Number of Clusters')
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
    
    for feature in specific_categorical_features:
        print(f"\nDistribution of {feature} across Clusters:")
        print(df.groupby(['Cluster', feature]).size().unstack(fill_value=0))
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='Cluster', hue=feature)
        plt.title(f'Distribution of {feature} across Clusters')
        plt.show()

def main():
    filepath = './retraction_before_modelling.csv'
    df = load_and_clean_data(filepath)
    print(df.shape)
    columns_to_analyze = ['Subject', 'Journal', 'Publisher', 'Country', 'ArticleType', 'Reason']
    df = preprocess_columns(df, columns_to_analyze)
    print(df.shape)
    df = perform_clustering(df, n_clusters=3)
    
    # Save the DataFrame after preprocessing and clustering
    df.to_csv('preprocessed_data_with_clusters.csv', index=False)
    print("Preprocessed data with cluster labels saved to 'preprocessed_data_with_clusters.csv'")
    print(df.columns)
    
    for feature in columns_to_analyze:
        df = filter_top_categories(df, feature)
    
    analyze_and_plot(df)
    # print(df.head())
    print(df.shape)
if __name__ == "__main__":
    main()

