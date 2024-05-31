import os
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Ensure the results/figures directory exists
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Load the data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        #print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Drop specified columns, duplicates, and rows with null 'Paywalled' values
def clean_data(df, columns_to_drop):
    initial_shape = df.shape
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['Paywalled'], inplace=True)
    df['Reason'] = df['Reason'].str.replace('+','',regex=False)

    final_shape = df.shape
    return df

# Convert date columns to datetime and create new columns for time difference and ratios
def transform_dates_and_ratios(df):
    df['OriginalPaperDate'] = pd.to_datetime(df['OriginalPaperDate'], format='%d/%m/%Y', errors='coerce')
    df['RetractionDate'] = pd.to_datetime(df['RetractionDate'], format='%d/%m/%Y', errors='coerce')
    df['TimeDifference_Days'] = (df['RetractionDate'] - df['OriginalPaperDate']).dt.days.replace(0, 1)
    df['CitationCount'].replace(0, 1.1, inplace=True)
    df['Retraction_to_Citation_Ratio'] = df['TimeDifference_Days'] / df['CitationCount']
    df['Citation_to_Retraction_Ratio'] = df['CitationCount'] / df['TimeDifference_Days']
    return df

# Transform the Subject column and apply transformation
def transform_and_apply_subject(df):
    df['Subject'] = df['Subject'].apply(lambda x: ', '.join(sorted(set(re.findall(r'\((.*?)\)', str(x))))))
    unique_subjects = set()
    df['Subject'].apply(lambda x: unique_subjects.update(x.split(', ')))
    return df

# Count occurrences of each individual entry within the cells of a column
def count_occurrences(df, column):
    occurrences = {}
    unique_values = set()
    
    for cell in df[column].dropna():
        for entry in re.split(r'[;,]', cell.strip()):
            entry = entry.strip()
            if entry:
                occurrences[entry] = occurrences.get(entry, 0) + 1
                unique_values.add(entry)
    
    return occurrences

# Plot the top 10 most frequently occurring entries for a column
def plot_top_10(df, column):
    occurrences = count_occurrences(df, column)
    sorted_occurrences = dict(sorted(occurrences.items(), key=lambda item: item[1], reverse=True)[:10])
    
    for entry, count in sorted_occurrences.items():
        print(f"{entry}: {count}")
    
    colors = plt.cm.tab10(np.arange(len(sorted_occurrences)))
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(sorted_occurrences.keys(), sorted_occurrences.values(), color=colors)
    plt.ylabel('Frequency')
    plt.title(f'Top 10 Most Frequently Occurring Entries in {column}')
    plt.xticks([])

    plt.legend(bars, sorted_occurrences.keys(), loc='upper right').get_frame().set_alpha(0.5)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.normpath(os.path.join(base_dir, '../../results/figures'))
    ensure_directory_exists(figures_dir)
    
    plt.savefig(os.path.join(figures_dir, f'top_10_{column}.png'))
    plt.show()

# Display network graph
def display_network_graph(df, column_name, top_n=None, weighted_degree_thresh=1, drop_thresh=5):
    all_vals = ";".join(list(df[column_name].dropna())).split(";")
    all_vals = [vv.strip() for vv in all_vals if vv not in ["", "Unknown", "unavailable", "Unavailable", "No affiliation available"]]
    all_vals_series = pd.Series(all_vals)
    top_rank = all_vals_series.value_counts().sort_values(ascending=False)
    
    if top_n is not None:
        more_than_1 = list(top_rank.head(top_n).index)
    else:
        more_than_1 = list(top_rank.index)
    
    if len(more_than_1) == 0:
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
        return

    G = nx.from_pandas_adjacency(aa_cor_mat_reduced)

    high_degree_nodes = [node for node, degree in G.degree(weight="weight") if degree > weighted_degree_thresh]
    G = G.subgraph(high_degree_nodes)
    
    if len(high_degree_nodes) == 0:
        return

    degrees = dict(G.degree())
    weighted_degrees = dict(G.degree(weight='weight'))
    highest_weighted_degree_node = max(weighted_degrees, key=weighted_degrees.get)
    
    edge_weights = [np.log(data['weight']) for _, _, data in G.edges(data=True)]
    normalized_edge_weights = edge_weights
    degs = [d[1]**1.2 for d in G.degree(weight="weight")]

    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=1.8)
    nx.draw(G, with_labels=True, pos=pos, font_weight='bold', edge_color=normalized_edge_weights, width=normalized_edge_weights, edge_cmap=plt.cm.Reds, edge_vmin=0, edge_vmax=4, node_size=degs, node_color=degs, cmap=plt.cm.autumn)
    plt.title(f"Network of Co-occurring {column_name}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.normpath(os.path.join(base_dir, '../../results/figures'))
    ensure_directory_exists(figures_dir)
    
    plt.savefig(os.path.join(figures_dir, f'network_graph_{column_name}.png'))
    plt.show()

# Frequency of Paywalled values
def plot_paywalled_frequency(df):
    retractions_by_paywalled = df['Paywalled'].value_counts()
    colors = plt.cm.tab10(np.arange(len(retractions_by_paywalled)))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(retractions_by_paywalled.index, retractions_by_paywalled.values, color=colors, alpha=0.7)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Paywalled')
    ax.set_title('Frequency of Unique Values in Paywalled Column')
    ax.set_xticklabels(retractions_by_paywalled.index, rotation=0)
    
    for i in range(len(retractions_by_paywalled)):
        ax.text(i, retractions_by_paywalled[i] + 5, str(retractions_by_paywalled[i]), ha='center', color='black')
    
    ax.legend(bars, retractions_by_paywalled.index, loc='upper right').get_frame().set_alpha(0.5)

    plt.grid(True)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.normpath(os.path.join(base_dir, '../../results/figures'))
    ensure_directory_exists(figures_dir)
    
    plt.savefig(os.path.join(figures_dir, 'paywalled_frequency.png'))
    plt.show()

# Distribution Analysis
def plot_distribution_analysis(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_filtered = df.dropna(subset=['Retraction_to_Citation_Ratio', 'Citation_to_Retraction_Ratio'])

    df_filtered[['CitationCount', 'TimeDifference_Days', 'Retraction_to_Citation_Ratio', 'Citation_to_Retraction_Ratio']].hist(bins=30, figsize=(15, 10))
    plt.tight_layout()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.normpath(os.path.join(base_dir, '../../results/figures'))
    ensure_directory_exists(figures_dir)
    
    plt.savefig(os.path.join(figures_dir, 'distribution_analysis.png'))
    plt.show()

    plt.figure(figsize=(15, 10))
    for i, column in enumerate(['CitationCount', 'TimeDifference_Days', 'Retraction_to_Citation_Ratio', 'Citation_to_Retraction_Ratio'], 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x=df_filtered[column])
        plt.title(f'Box plot of {column}')
    plt.tight_layout()
    
    plt.savefig(os.path.join(figures_dir, 'box_plots.png'))
    plt.show()

# Summary Statistics
def display_summary_statistics(df):
    summary_stats = df.describe()
    print("Summary Statistics:\n", summary_stats)

# Correlation Analysis
def plot_correlation_matrix(df):
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.normpath(os.path.join(base_dir, '../../results/figures'))
    ensure_directory_exists(figures_dir)
    
    plt.savefig(os.path.join(figures_dir, 'correlation_matrix_heatmap.png'))
    plt.show()

# Time Series Analysis
def plot_time_series_analysis(df):
    df_filtered = df[df['RetractionDate'].dt.year < 2024]
    df_filtered = df_filtered[df['OriginalPaperDate'].dt.year < 2023]

    df_filtered['Year_Retraction'] = df_filtered['RetractionDate'].dt.year
    retractions_by_year = df_filtered['Year_Retraction'].value_counts().sort_index()

    df_filtered['Year_Publication'] = df_filtered['OriginalPaperDate'].dt.year
    publications_by_year = df_filtered['Year_Publication'].value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    plt.plot(publications_by_year.index, publications_by_year.values, label='Publications', color='cyan', marker='o')
    plt.plot(retractions_by_year.index, retractions_by_year.values, label='Retractions', color='blue', marker='o')
    plt.title('Number of Publications and Retractions Over Time')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.normpath(os.path.join(base_dir, '../../results/figures'))
    ensure_directory_exists(figures_dir)
    
    plt.savefig(os.path.join(figures_dir, 'publications_retractions_over_time.png'))
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
    
    plt.savefig(os.path.join(figures_dir, 'publications_retractions_recent_years.png'))
    plt.show()

# Time Series Analysis by categorical columns
def plot_time_series_by_category(df, column, top_n):
    occurrences = count_occurrences(df, column)
    top_values = sorted(occurrences, key=occurrences.get, reverse=True)[:top_n]

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
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.normpath(os.path.join(base_dir, '../../results/figures'))
    ensure_directory_exists(figures_dir)
    
    plt.savefig(os.path.join(figures_dir, f'time_series_{column}.png'))
    plt.show()

# Plot retractions and citations by top 5 category values
def plot_retractions_and_citations(df, category):
    occurrences = count_occurrences(df, category)
    top_values = sorted(occurrences, key=occurrences.get, reverse=True)[:5]
    df_top = df[df[category].apply(lambda x: any(val in str(x) for val in top_values))]
    
    retractions_by_category = df_top[category].apply(lambda x: [val for val in top_values if val in str(x)]).explode().value_counts()
    citations_by_category = {value: df_top[df_top[category].str.contains(value, na=False, regex=False)]['CitationCount'].sum() for value in top_values}
    citations_by_category = pd.Series(citations_by_category)
    
    sorted_indices = retractions_by_category.sort_values().index
    retractions_by_category = retractions_by_category[sorted_indices]
    citations_by_category = citations_by_category[sorted_indices]
    
    colors_citations = 'blue'
    colors_retractions = 'red'
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bar_width = 0.35
    index = np.arange(len(top_values))
    
    bars1 = ax.bar(index, citations_by_category.values, bar_width, color=colors_citations, alpha=0.6, label='Total Citations')
    bars2 = ax.bar(index + bar_width, retractions_by_category.values, bar_width, color=colors_retractions, alpha=0.6, label='Total Retractions')

    ax.set_ylabel('Count')
    ax.set_title(f'Total Number of Retractions and Total Citations by Top 5 {category} (1990-2023')
    
    def shorten_label(label):
        words = label.split()
        if len(words) > 3:
            return ' '.join(words[:3]) + '...'
        else:
            return label

    labels = [shorten_label(val) for val in retractions_by_category.index]
    
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 5, f'{int(yval)}', ha='center', va='bottom', color='black')

    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 5, f'{int(yval)}', ha='center', va='bottom', color='black')

    ax.legend(loc='upper right', title='Type')

    plt.grid(True)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.normpath(os.path.join(base_dir, '../../results/figures'))
    ensure_directory_exists(figures_dir)
    
    plt.savefig(os.path.join(figures_dir, f'retractions_and_citations_{category}.png'))
    plt.show()


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_world_map_analysis(file_path):
    country_data = pd.read_csv(file_path)
    total_frequency = country_data['Frequency'].sum()
    country_data['Percentage'] = (country_data['Frequency'] / total_frequency) * 100

    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    bins = pd.cut(country_data['Frequency'], bins=10, labels=color_palette, include_lowest=True)
    country_data['color'] = bins
    country_data['hover_text'] = country_data.apply(lambda row: f"Country: {row['Country']}<br>Freq: {row['Frequency']}<br>{row['Percentage']:.2f}%", axis=1)
    
    fig = px.choropleth(country_data, locations='Country', locationmode='country names', color='Frequency', hover_name='Country',
                        hover_data={'Frequency': True, 'Percentage': True, 'color': False}, title='World Map with Country Frequencies',
                        color_continuous_scale=color_palette)
    
    top_5_countries = country_data.nlargest(5, 'Frequency')
    for _, row in top_5_countries.iterrows():
        fig.add_trace(go.Scattergeo(locationmode='country names', locations=[row['Country']], text=row['Country'], mode='text', showlegend=False,
                                    textfont=dict(size=16, color='black')))
    
    fig.update_layout(width=1600, height=1200, coloraxis_colorbar=dict(title='Frequency'))
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.normpath(os.path.join(base_dir, '../../results/figures'))
    ensure_directory_exists(figures_dir)
    
    output_file_path = os.path.join(figures_dir, 'worldmap.png')
    fig.write_image(output_file_path)  # Save the figure as a PNG file
    fig.show()  # Display the figure


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

    colors = sns.color_palette("husl", len(labels))

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops=dict(color="w")
    )

    for text in texts:
        text.set_color('black')
        text.set_fontsize(10)

    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(10)

    ax.set_title(f'Top 3 {column} when Country is China')
    plt.tight_layout()
    plt.show()

def save_data(df, filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.normpath(os.path.join(base_dir, '../../data/processed', filename))
    df.to_csv(output_file_path, index=False)
    #print(f"Data saved to {output_file_path}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = 'retractions35215.csv'
    file_path = os.path.normpath(os.path.join(base_dir, '../../data/raw', file_name))
    
    df = load_data(file_path)
    if df is None:
        return

    columns_to_drop = ['Record ID', 'URLS', 'OriginalPaperDOI', 'RetractionDOI', 'RetractionPubMedID', 
                       'OriginalPaperPubMedID', 'Title', 'Institution', 'Notes']
    df = clean_data(df, columns_to_drop)
    df = transform_dates_and_ratios(df)
    df = transform_and_apply_subject(df)

    save_data(df, 'retraction_before_modelling.csv')
    
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

    correlation_matrix = df.corr()
    print("Correlation Matrix:")
    print(correlation_matrix)
    print()

    print("Correlation Matrix Heatmap:")
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix Heatmap')
    plt.show()

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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = 'country.csv'
    file_path = os.path.normpath(os.path.join(base_dir, '../../data/raw', file_name))
    
    plot_world_map_analysis(file_path)


def main_top_3_pie_china():
    df = main()
    if df is None:
        return
    
    df_china = filter_country(df, 'China')
    columns_to_plot = ['Reason', 'Publisher', 'Subject']
    for column in columns_to_plot:
        plot_top_3_pie(df_china, column)

if __name__ == '__main__':
    # main_top_10()
    # main_network_graphs()
    #main_paywalled_frequency()
    # main_distribution_analysis()
    # main_summary_statistics()
    # main_correlation_matrix()
    # main_time_series_analysis()
    # main_time_series_by_category()
    # main_retractions_and_citations()
    main_world_map_analysis()
    # main_top_3_pie_china()
    
    # For demonstration, you can uncomment the following line to execute the main function
    # df_transformed = main()
    # if df_transformed is not None:
    #     print("Column names after transformations:", df_transformed.columns.tolist())
