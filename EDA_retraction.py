import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Load the data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        # print(f"Data shape: {df.shape}")
        # print(f"Column names: {df.columns.tolist()}")
        # print(df.head())
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Drop specified columns, duplicates, and rows with null 'Paywalled' values
def clean_data(df, columns_to_drop):
    # print("\nCleaning data...")
    initial_shape = df.shape
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['Paywalled'], inplace=True)
    df['Reason'] = df['Reason'].str.replace('+','',regex=False)

    final_shape = df.shape
    # print(f"Initial shape: {initial_shape}, Final shape after cleaning: {final_shape}")
    # print(df.head())
    return df

# Convert date columns to datetime and create new columns for time difference and ratios
def transform_dates_and_ratios(df):
    # print("\nTransforming date columns and creating ratio columns...")
    df['OriginalPaperDate'] = pd.to_datetime(df['OriginalPaperDate'], format='%d/%m/%Y', errors='coerce')
    df['RetractionDate'] = pd.to_datetime(df['RetractionDate'], format='%d/%m/%Y', errors='coerce')
    df['TimeDifference_Days'] = (df['RetractionDate'] - df['OriginalPaperDate']).dt.days.replace(0, 1)
    df['CitationCount'].replace(0, 1.1, inplace=True)
    df['Retraction_to_Citation_Ratio'] = df['TimeDifference_Days'] / df['CitationCount']
    df['Citation_to_Retraction_Ratio'] = df['CitationCount'] / df['TimeDifference_Days']
    # print(df[['OriginalPaperDate', 'RetractionDate', 'TimeDifference_Days', 'CitationCount', 'Retraction_to_Citation_Ratio', 'Citation_to_Retraction_Ratio']].head())
    return df

# Transform the Subject column and apply transformation
def transform_and_apply_subject(df):
    # print("\nTransforming Subject column...")
    
    # Apply transformation to standardize the 'Subject' column
    df['Subject'] = df['Subject'].apply(lambda x: ', '.join(sorted(set(re.findall(r'\((.*?)\)', str(x))))))
    
    # Print the transformed 'Subject' column
    # print(df['Subject'].head())
    
    # Extract and count unique subjects
    unique_subjects = set()
    df['Subject'].apply(lambda x: unique_subjects.update(x.split(', ')))
    
    # print("\nUnique occurrences of subjects:")
    # for subject in sorted(unique_subjects):
    #     print(subject)
    
    return df


# Count occurrences of each individual entry within the cells of a column
def count_occurrences(df, column):
    # print(f"\nCounting occurrences in {column} column...")
    occurrences = {}
    unique_values = set()
    
    for cell in df[column].dropna():
        # Split the entries and strip any extra spaces from each entry
        for entry in re.split(r'[;,]', cell.strip()):
            entry = entry.strip()
            if entry:
                occurrences[entry] = occurrences.get(entry, 0) + 1
                unique_values.add(entry)
    
    # Sort the occurrences and print the top 10
    top_occurrences = dict(sorted(occurrences.items(), key=lambda item: item[1], reverse=True)[:10])
    # print(f"Top occurrences: {top_occurrences}")
    
    # Print the number of unique values and the unique values themselves
    # print(f"Number of unique values in {column}: {len(unique_values)}")
    # print(f"Unique values in {column}: {sorted(unique_values)}")
    
    return occurrences

# Plot the top 10 most frequently occurring entries for a column
def plot_top_10(df, column):
    occurrences = count_occurrences(df, column)
    sorted_occurrences = dict(sorted(occurrences.items(), key=lambda item: item[1], reverse=True)[:10])
    
    # Print the table for top 10 occurrences
    # print(f"\nTop 10 most frequently occurring entries in {column}:")
    for entry, count in sorted_occurrences.items():
        print(f"{entry}: {count}")
    
    # Generate a list of distinct colors for each bar
    colors = plt.cm.tab10(np.arange(len(sorted_occurrences)))
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(sorted_occurrences.keys(), sorted_occurrences.values(), color=colors)
    plt.ylabel('Frequency')
    plt.title(f'Top 10 Most Frequently Occurring Entries in {column}')
    plt.xticks([])  # Remove x-axis labels

    # Create legend with corresponding colors and a transparent background
    plt.legend(bars, sorted_occurrences.keys(), loc='upper right').get_frame().set_alpha(0.5)

    plt.savefig(f'top_10_{column}.png')
    plt.show()

# Display network graph
def display_network_graph(df, column_name, top_n=None, weighted_degree_thresh=1, drop_thresh=5):
    # print(f"\nCreating network graph for {column_name}...")
    
    all_vals = ";".join(list(df[column_name].dropna())).split(";")
    all_vals = [vv.strip() for vv in all_vals if vv not in ["", "Unknown", "unavailable", "Unavailable", "No affiliation available"]]
    all_vals_series = pd.Series(all_vals)
    top_rank = all_vals_series.value_counts().sort_values(ascending=False)
    # print(f"Top values: {top_rank.head(10)}")
    
    if top_n is not None:
        more_than_1 = list(top_rank.head(top_n).index)
    else:
        more_than_1 = list(top_rank.index)
    
    if len(more_than_1) == 0:
        # print(f"No entries for {column_name}. Skipping...")
        return

    aa_cor_mat = pd.DataFrame(np.zeros((len(more_than_1), len(more_than_1))), columns=more_than_1, index=more_than_1)

    column_lists = list(df[column_name].dropna().str.split(";"))
    for cl in column_lists:
        cl = [item.strip() for item in cl]
        for ii in range(len(cl)):
            if cl[ii] not in more_than_1:
                continue
            for jj in range(ii + 1, len(cl)):
                if cl[jj] not in more_than_1:
                    continue
                aa_cor_mat.loc[cl[ii], cl[jj]] += 1
                aa_cor_mat.loc[cl[jj], cl[ii]] += 1
    
    aa_cor_mat_reduced = aa_cor_mat.copy()
    for au in aa_cor_mat.columns:
        if aa_cor_mat[au].sum() < drop_thresh:
            aa_cor_mat_reduced = aa_cor_mat_reduced.drop(columns=[au])
            aa_cor_mat_reduced = aa_cor_mat_reduced.drop(index=[au])
    
    if aa_cor_mat_reduced.shape[0] == 0 or aa_cor_mat_reduced.shape[1] == 0:
        # print(f"Empty reduced correlation matrix for {column_name}. Skipping...")
        return

    G = nx.from_pandas_adjacency(aa_cor_mat_reduced)

    high_degree_nodes = [node for node, degree in G.degree(weight="weight") if degree > weighted_degree_thresh]
    G = G.subgraph(high_degree_nodes)
    
    if len(high_degree_nodes) == 0:
        # print(f"No high degree nodes for {column_name} after threshold filtering. Skipping...")
        return

    degrees = dict(G.degree())
    weighted_degrees = dict(G.degree(weight='weight'))
    highest_weighted_degree_node = max(weighted_degrees, key=weighted_degrees.get)
    
    edge_weights = [np.log(data['weight']) for _, _, data in G.edges(data=True)]
    normalized_edge_weights = edge_weights
    degs = [d[1]**1.2 for d in G.degree(weight="weight")]

    # print(f"\nNodes with highest weighted degrees in {column_name} network graph:")
    # for node, degree in sorted(weighted_degrees.items(), key=lambda item: item[1], reverse=True)[:10]:
    #     print(f"{node}: {degree}")
    
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=1.8)
    nx.draw(G, with_labels=True, pos=pos, font_weight='bold', edge_color=normalized_edge_weights, width=normalized_edge_weights, edge_cmap=plt.cm.Reds, edge_vmin=0, edge_vmax=4, node_size=degs, node_color=degs, cmap=plt.cm.autumn)
    plt.title(f"Network of Co-occurring {column_name}")
    plt.savefig(f'network_graph_{column_name}.png')
    plt.show()  

# Frequency of Paywalled values
def plot_paywalled_frequency(df):
    # print("\nPlotting Paywalled frequency...")
    retractions_by_paywalled = df['Paywalled'].value_counts()
    # print("Paywalled Frequency Count:")
    # print(retractions_by_paywalled)
    colors = plt.cm.tab10(np.arange(len(retractions_by_paywalled)))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(retractions_by_paywalled.index, retractions_by_paywalled.values, color=colors, alpha=0.7)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Paywalled')
    ax.set_title('Frequency of Unique Values in Paywalled Column')
    ax.set_xticklabels(retractions_by_paywalled.index, rotation=0)
    
    for i in range(len(retractions_by_paywalled)):
        ax.text(i, retractions_by_paywalled[i] + 5, str(retractions_by_paywalled[i]), ha='center', color='black')
    
    # Create legend with corresponding colors and a transparent background
    ax.legend(bars, retractions_by_paywalled.index, loc='upper right').get_frame().set_alpha(0.5)

    plt.grid(True)
    plt.savefig('paywalled_frequency.png')
    plt.show()

# Distribution Analysis
def plot_distribution_analysis(df):
    # print("\nPlotting distribution analysis...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_filtered = df.dropna(subset=['Retraction_to_Citation_Ratio', 'Citation_to_Retraction_Ratio'])
    
    # print("Distribution of Retraction to Citation Ratio:")
    # print(df_filtered['Retraction_to_Citation_Ratio'].describe())
    
    # print("Distribution of Citation to Retraction Ratio:")
    # print(df_filtered['Citation_to_Retraction_Ratio'].describe())
    
    # print("Distribution of Mean duration:")
    # print(df_filtered['TimeDifference_Days'].describe())
    
    # print("Distribution of CitationCount:")
    # print(df_filtered['CitationCount'].describe())

    df_filtered[['CitationCount', 'TimeDifference_Days', 'Retraction_to_Citation_Ratio', 'Citation_to_Retraction_Ratio']].hist(bins=30, figsize=(15, 10))
    plt.tight_layout()
    plt.savefig('distribution_analysis.png')
    plt.show()

    plt.figure(figsize=(15, 10))
    for i, column in enumerate(['CitationCount', 'TimeDifference_Days', 'Retraction_to_Citation_Ratio', 'Citation_to_Retraction_Ratio'], 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x=df_filtered[column])
        plt.title(f'Box plot of {column}')
        # print(f"\nBox plot statistics for {column}:")
        # print(df_filtered[column].describe())
    plt.tight_layout()
    plt.savefig('box_plots.png')
    plt.show()

# Summary Statistics
def display_summary_statistics(df):
    # print("\nDisplaying summary statistics...")
    summary_stats = df.describe()
    # print("Summary Statistics:\n", summary_stats)

# Correlation Analysis
def plot_correlation_matrix(df):
    # print("\nPlotting correlation matrix...")
    correlation_matrix = df.corr()
    # print("Correlation matrix:\n", correlation_matrix)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.savefig('correlation_matrix_heatmap.png')
    plt.show()

# Time Series Analysis
def plot_time_series_analysis(df):
    # print("\nPlotting time series analysis...")
    df_filtered = df[df['RetractionDate'].dt.year < 2024]
    df_filtered = df_filtered[df['OriginalPaperDate'].dt.year < 2023]

    df_filtered['Year_Retraction'] = df_filtered['RetractionDate'].dt.year
    retractions_by_year = df_filtered['Year_Retraction'].value_counts().sort_index()

    df_filtered['Year_Publication'] = df_filtered['OriginalPaperDate'].dt.year
    publications_by_year = df_filtered['Year_Publication'].value_counts().sort_index()

    # print("Retractions by year:")
    # print(retractions_by_year)
    
    # print("Publications by year:")
    # print(publications_by_year)

    plt.figure(figsize=(10, 5))
    plt.plot(publications_by_year.index, publications_by_year.values, label='Publications', color='cyan', marker='o')
    plt.plot(retractions_by_year.index, retractions_by_year.values, label='Retractions', color='blue', marker='o')
    plt.title('Number of Publications and Retractions Over Time')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.savefig('publications_retractions_over_time.png')
    plt.show()

    common_recent_years = retractions_by_year.index.intersection(publications_by_year.index)
    common_recent_years = common_recent_years.sort_values()[-10:]

    retractions_recent = retractions_by_year.loc[common_recent_years]
    publications_recent = publications_by_year.loc[common_recent_years]

    plt.figure(figsize=(10, 5))
    plt.plot(publications_recent.index, publications_recent.values, label='Publications', color='cyan', marker='o')
    plt.plot(retractions_recent.index, retractions_recent.values, label='Retractions', color='blue', marker='o')
    plt.title('Number of Publications and Retractions Over the Past 10 Years')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.savefig('publications_retractions_recent_years.png')
    plt.show()

# Time Series Analysis by categorical columns
def plot_time_series_by_category(df, column, top_n):
    occurrences = count_occurrences(df, column)
    top_values = sorted(occurrences, key=occurrences.get, reverse=True)[:top_n]
    
    # print(f"\nTop {top_n} values in {column} for time series analysis:")
    # print(top_values)
    
    fig, ax_main = plt.subplots(figsize=(12, 6))
    
    for value in top_values:
        df_value_filtered = df[df[column].str.contains(value, na=False, regex=False)].copy()
        df_value_filtered.loc[:, 'Year_Retraction'] = df_value_filtered['RetractionDate'].dt.year
        retractions_by_year = df_value_filtered['Year_Retraction'].value_counts().sort_index()
        ax_main.plot(retractions_by_year.index, retractions_by_year.values, marker='o', label=value)
    
    ax_main.set_title(f'Retractions Over Time for Top {top_n} {column}')
    ax_main.set_xlabel('Year')
    ax_main.set_ylabel('Count')
    ax_main.legend()
    ax_main.grid(True)

    if column != 'Author':
        ax_inset = inset_axes(ax_main, width="40%", height="40%", loc="center left")
        recent_years = retractions_by_year.index.sort_values()[-5:]
        
        for value in top_values:
            df_value_filtered = df[df[column].str.contains(value, na=False, regex=False)].copy()
            df_value_filtered.loc[:, 'Year_Retraction'] = df_value_filtered['RetractionDate'].dt.year
            retractions_by_year = df_value_filtered['Year_Retraction'].value_counts().sort_index()
            common_recent_years = recent_years.intersection(retractions_by_year.index)
            retractions_recent = retractions_by_year.loc[common_recent_years]
            ax_inset.plot(retractions_recent.index, retractions_recent.values, marker='o', label=value)
        
        ax_inset.yaxis.set_label_position("right")
        ax_inset.yaxis.tick_right()
        ax_inset.set_title('Last 5 Years')
        ax_inset.set_xlabel('Year')
        ax_inset.set_ylabel('Count')
        ax_inset.grid(True)
    
    plt.savefig(f'time_series_{column}.png')
    plt.show()

# Plot retractions and citations by top 5 category values
def plot_retractions_and_citations(df, category):
    occurrences = count_occurrences(df, category)
    top_values = sorted(occurrences, key=occurrences.get, reverse=True)[:5]
    df_top = df[df[category].apply(lambda x: any(val in str(x) for val in top_values))]
    
    retractions_by_category = df_top[category].apply(lambda x: [val for val in top_values if val in str(x)]).explode().value_counts()
    citations_by_category = {value: df_top[df_top[category].str.contains(value, na=False, regex=False)]['CitationCount'].sum() for value in top_values}
    citations_by_category = pd.Series(citations_by_category)
    
    # Sort the values by retractions in ascending order
    sorted_indices = retractions_by_category.sort_values().index
    retractions_by_category = retractions_by_category[sorted_indices]
    citations_by_category = citations_by_category[sorted_indices]
    
    # Print the table for retractions and citations
    # print(f"\nRetractions and citations for top 5 {category} values:")
    # print("Retractions:")
    # print(retractions_by_category)
    # print("Citations:")
    # print(citations_by_category)
    
    # Generate colors for each type of bar
    colors_citations = 'blue'
    colors_retractions = 'red'
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bar_width = 0.35
    index = np.arange(len(top_values))
    
    bars1 = ax.bar(index, citations_by_category.values, bar_width, color=colors_citations, alpha=0.6, label='Total Citations')
    bars2 = ax.bar(index + bar_width, retractions_by_category.values, bar_width, color=colors_retractions, alpha=0.6, label='Total Retractions')

    ax.set_ylabel('Count')
    ax.set_title(f'Total Number of Retractions and Total Citations by Top 5 {category} (1990-2023')
    
    # Create combined labels with up to 4 words
    def shorten_label(label):
        words = label.split()
        if len(words) > 3:
            return ' '.join(words[:3]) + '...'
        else:
            return label

    labels = [shorten_label(val) for val in retractions_by_category.index]
    
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Embed labels in the plot
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 5, f'{int(yval)}', ha='center', va='bottom', color='black')

    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 5, f'{int(yval)}', ha='center', va='bottom', color='black')

    ax.legend(loc='upper right', title='Type')

    plt.grid(True)
    plt.savefig(f'retractions_and_citations_{category}.png')
    plt.show()

# World Map Analysis
def plot_world_map_analysis(file_path):
    country_data = pd.read_csv(file_path)
    total_frequency = country_data['Frequency'].sum()
    country_data['Percentage'] = (country_data['Frequency'] / total_frequency) * 100
    
    # Print the country data for interpretation
    # print("\nCountry data for world map analysis:")
    # print(country_data.head())

    # Define color palette and binning
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    bins = pd.cut(country_data['Frequency'], bins=10, labels=color_palette, include_lowest=True)
    country_data['color'] = bins
    country_data['hover_text'] = country_data.apply(lambda row: f"Country: {row['Country']}<br>Freq: {row['Frequency']}<br>{row['Percentage']:.2f}%", axis=1)
    
    # Plot world map
    fig = px.choropleth(country_data, locations='Country', locationmode='country names', color='Frequency', hover_name='Country',
                        hover_data={'Frequency': True, 'Percentage': True, 'color': False}, title='World Map with Country Frequencies',
                        color_continuous_scale=color_palette)
    
    top_5_countries = country_data.nlargest(5, 'Frequency')
    for _, row in top_5_countries.iterrows():
        fig.add_trace(go.Scattergeo(locationmode='country names', locations=[row['Country']], text=row['Country'], mode='text', showlegend=False,
                                    textfont=dict(size=16, color='black')))
    
    fig.update_layout(width=1600, height=1200, coloraxis_colorbar=dict(title='Frequency'))
    fig.write_image("world_map_frequencies.pdf")
    fig.show()

# Filter the data for entries where the country is China
def filter_country(df, country):
    return df[df['Country'] == country]

# Function to wrap text into multiple lines with a maximum of three words per line
def wrap_text(text, max_words_per_line=3):
    words = text.split()
    wrapped_text = '\n'.join(
        [' '.join(words[i:i + max_words_per_line]) for i in range(0, len(words), max_words_per_line)]
    )
    return wrapped_text

# Function to plot top 3 values for a given column as a pie chart with an "Other" category
def plot_top_3_pie(df, column):
    value_counts = df[column].value_counts()
    top_3 = value_counts.nlargest(3)
    other_count = value_counts.iloc[3:].sum()

    labels = [wrap_text(label) for label in top_3.index] + ['Other']
    sizes = list(top_3.values) + [other_count]

    # Define a color palette
    colors = sns.color_palette("husl", len(labels))

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops=dict(color="w")
    )

    # Customizing text inside the slices
    for text in texts:
        text.set_color('black')
        text.set_fontsize(10)

    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(10)

    ax.set_title(f'Top 3 {column} when Country is China')
    plt.tight_layout()
    plt.show()

def main():
    file_path = './retractions35215.csv'
    df = load_data(file_path)
    if df is None:
        return

    # Drop unnecessary columns and clean data
    columns_to_drop = ['Record ID', 'URLS', 'OriginalPaperDOI', 'RetractionDOI', 'RetractionPubMedID', 
                       'OriginalPaperPubMedID', 'Title', 'Institution', 'Notes']
    df = clean_data(df, columns_to_drop)

    # Data Cleaning and Transformation
    df = transform_dates_and_ratios(df)
    df = transform_and_apply_subject(df)
    print('File saving:')
    save_data(df, 'file_before_modelling.csv')
    return df



def main_top_10():
    df = main()
    if df is None:
        return
    
    columns_to_analyze = ['Subject', 'Journal', 'Publisher', 'Country', 'ArticleType', 'Reason', 'Author']
    for column in columns_to_analyze:
        plot_top_10(df, column)

def main_network_graphs():
    df = main()
    if df is None:
        return
    
    params = {
        'Reason': {'top_n': 20, 'weighted_degree_thresh': 1, 'drop_thresh': 5},
        'Author': {'top_n': 10, 'weighted_degree_thresh': 1, 'drop_thresh': 2},
        'Subject': {'top_n': 100, 'weighted_degree_thresh': 1, 'drop_thresh': 1},
        'Country': {'top_n': 50, 'weighted_degree_thresh': 1, 'drop_thresh': 2}
    }
    for column, param in params.items():
        display_network_graph(df, column, **param)

def main_paywalled_frequency():
    df = main()
    if df is None:
        return
    plot_paywalled_frequency(df)

def main_distribution_analysis():
    df = main()
    if df is None:
        return
    plot_distribution_analysis(df)

def main_summary_statistics():
    df = main()
    if df is None:
        return
    display_summary_statistics(df)

def main_correlation_matrix():
    df = main()
    if df is None:
        return
    plot_correlation_matrix(df)

def main_time_series_analysis():
    df = main()
    if df is None:
        return
    plot_time_series_analysis(df)

def main_time_series_by_category():
    df = main()
    if df is None:
        return
    
    top_n_values = {'Subject': 5, 'Journal': 5, 'Author': 5, 'Publisher': 5, 'Country': 5, 'ArticleType': 5, 'Reason': 5}
    for column, top_n in top_n_values.items():
        plot_time_series_by_category(df, column, top_n)

def main_retractions_and_citations():
    df = main()
    if df is None:
        return
    
    df_no_paywall = df[df['Paywalled'] == 'No']
    for category in ['Subject', 'Journal', 'Publisher', 'Country', 'ArticleType', 'Reason']:
        plot_retractions_and_citations(df_no_paywall, category)

def main_world_map_analysis():
    plot_world_map_analysis('country.csv')

def main_top_3_pie_china():
    df = main()
    if df is None:
        return
    
    df_china = filter_country(df, 'China')
    columns_to_plot = ['Reason', 'Publisher', 'Subject']
    for column in columns_to_plot:
        plot_top_3_pie(df_china, column)
def save_data(df, filename):
    df.to_csv('retraction_before_modelling.csv', index=False)
    print(f"Data saved to {filename}")


if __name__ == '__main__':
    # Choose which main function to run based on the analysis you want to perform
    # Uncomment the appropriate function call below to execute it

    #main_top_10()
    # main_network_graphs()
    # main_paywalled_frequency()
    # main_distribution_analysis()
    # main_summary_statistics()
    # main_correlation_matrix()
    # main_time_series_analysis()
    # main_time_series_by_category()
    # main_retractions_and_citations()
    # main_world_map_analysis()
    main_top_3_pie_china()
    
    # For demonstration, you can uncomment the following line to execute the main function
    # df_transformed = main()
    # if df_transformed is not None:
    #     print("Column names after transformations:", df_transformed.columns.tolist())
