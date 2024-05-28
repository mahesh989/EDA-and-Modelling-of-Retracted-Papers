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
    return pd.read_csv(file_path)

# Drop specified columns
def drop_columns(df, columns_to_drop):
    df.drop(columns=columns_to_drop, inplace=True)
    return df

# Drop rows with null values in the 'Paywalled' column
def drop_null_paywalled(df):
    df.dropna(subset=['Paywalled'], inplace=True)
    return df

# Convert date columns to datetime
def convert_dates(df):
    df['OriginalPaperDate'] = pd.to_datetime(df['OriginalPaperDate'], dayfirst=True, errors='coerce')
    df['RetractionDate'] = pd.to_datetime(df['RetractionDate'], dayfirst=True, errors='coerce')
    return df

# Create new columns for the time difference between OriginalPaperDate and RetractionDate
def create_time_difference(df):
    df['TimeDifference_Days'] = (df['RetractionDate'] - df['OriginalPaperDate']).dt.days
    return df

# Create a new column for retraction to citation ratio
def create_retraction_to_citation_ratio(df):
    df['Retraction_to_Citation_Ratio'] = df.apply(
        lambda row: row['TimeDifference_Days'] / row['CitationCount'] if row['CitationCount'] != 0 else float('inf'), axis=1)
    return df

# Create a new column for citation to retraction ratio
def create_citation_to_retraction_ratio(df):
    df['Citation_to_Retraction_Ratio'] = df.apply(
        lambda row: row['CitationCount'] / row['TimeDifference_Days'] if row['TimeDifference_Days'] != 0 else float('inf'), axis=1)
    return df

# Transform the Subject column
def transform_subject(subject):
    codes = re.findall(r'\((.*?)\)', subject)
    return ', '.join(sorted(set(codes)))

def apply_transform_subject(df):
    df['Subject'] = df['Subject'].apply(transform_subject)
    return df

# Count occurrences of each individual entry within the cells of a column
def count_occurrences(df, column):
    occurrences = {}
    for cell in df[column].dropna():
        entries = re.split(r'[;,]', cell)
        for entry in entries:
            entry = entry.strip()
            if entry:
                if entry in occurrences:
                    occurrences[entry] += 1
                else:
                    occurrences[entry] = 1
    return occurrences

# Plot the top 10 most frequently occurring entries for a column
def plot_top_10(df, column):
    occurrences = count_occurrences(df, column)
    sorted_occurrences = dict(sorted(occurrences.items(), key=lambda item: item[1], reverse=True)[:10])
    
    # Generate a list of distinct colors for each bar
    colors = plt.cm.tab10(np.arange(len(sorted_occurrences)))
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(sorted_occurrences.keys(), sorted_occurrences.values(), color=colors)
    plt.ylabel('Frequency')
    plt.title(f'Top 10 Most Frequently Occurring Entries in {column}')
    plt.xticks([])  # Remove x-axis labels

    # Create legend with corresponding colors and a transparent background
    legend_labels = sorted_occurrences.keys()
    legend = plt.legend(bars, legend_labels, loc='upper right')
    legend.get_frame().set_alpha(0.5)  # Set legend background to be transparent

    plt.savefig(f'top_10_{column}.png')
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
        print(f"No entries for {column_name}. Skipping...")
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
        print(f"Empty reduced correlation matrix for {column_name}. Skipping...")
        return

    G = nx.from_pandas_adjacency(aa_cor_mat_reduced)

    high_degree_nodes = [node for node, degree in G.degree(weight="weight") if degree > weighted_degree_thresh]
    G = G.subgraph(high_degree_nodes)
    
    if len(high_degree_nodes) == 0:
        print(f"No high degree nodes for {column_name} after threshold filtering. Skipping...")
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
    plt.savefig(f'network_graph_{column_name}.png')
    plt.show()

# Frequency of Paywalled values
def plot_paywalled_frequency(df):
    retractions_by_paywalled = df['Paywalled'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    retractions_by_paywalled.plot(kind='bar', ax=ax, color='green', alpha=0.7)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Paywalled')
    ax.set_title('Frequency of Unique Values in Paywalled Column')
    ax.set_xticklabels(retractions_by_paywalled.index, rotation=0)
    for i in range(len(retractions_by_paywalled)):
        ax.text(i, retractions_by_paywalled[i] + 5, str(retractions_by_paywalled[i]), ha='center', color='black')
    plt.grid(True)
    plt.savefig('paywalled_frequency.png')
    plt.show()

# Distribution Analysis
def plot_distribution_analysis(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_filtered = df.dropna(subset=['Retraction_to_Citation_Ratio', 'Citation_to_Retraction_Ratio'])

    plt.figure(figsize=(15, 10))
    df_filtered[['CitationCount', 'TimeDifference_Days', 'Retraction_to_Citation_Ratio', 'Citation_to_Retraction_Ratio']].hist(bins=30)
    plt.tight_layout()
    plt.savefig('distribution_analysis.png')
    plt.show()

    plt.figure(figsize=(15, 10))
    for i, column in enumerate(['CitationCount', 'TimeDifference_Days', 'Retraction_to_Citation_Ratio', 'Citation_to_Retraction_Ratio'], 1):
        plt.subplot(2, 2, i)
        sns.boxplot(df_filtered[column])
        plt.title(f'Box plot of {column}')
    plt.tight_layout()
    plt.savefig('box_plots.png')
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
    plt.savefig('correlation_matrix_heatmap.png')
    plt.show()

# Time Series Analysis
def plot_time_series_analysis(df):
    df_filtered = df[df['RetractionDate'].dt.year < 2024]
    df_filtered = df_filtered[df_filtered['OriginalPaperDate'].dt.year < 2023]

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
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    retractions_by_category.plot(kind='bar', ax=ax1, color='blue', alpha=0.6, position=0, width=0.4, label='Total Retractions')
    ax1.set_ylabel('Total Retractions')
    ax1.set_xlabel(category)
    ax1.set_title(f'Total Number of Retractions and Total Citations by Top 5 {category} (1990-2023, Paywalled: No)')
    
    ax2 = ax1.twinx()
    citations_by_category.plot(kind='bar', ax=ax2, color='red', alpha=0.6, position=1, width=0.4, label='Total Citations')
    ax2.set_ylabel('Total Citations')
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.savefig(f'retractions_and_citations_{category}.png')
    plt.show()

# World Map Analysis
def plot_world_map_analysis(file_path):
    country_data = pd.read_csv(file_path)
    total_frequency = country_data['Frequency'].sum()
    country_data['Percentage'] = (country_data['Frequency'] / total_frequency) * 100
    color_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
        '#bcbd22', '#17becf', '#9edae5'
    ]
    above_22000_color = '#ff1493'
    max_frequency = country_data['Frequency'].max()
    bins = pd.cut(country_data['Frequency'], bins=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, max_frequency],
                  labels=['0-500', '501-1000', '1001-1500', '1501-2000', '2001-2500', '2501-3000', '3001-3500', '3501-4000', '4001-4500', f'4501-{max_frequency}'], include_lowest=True)
    bin_color_map = {
        '0-500': color_palette[0],
        '501-1000': color_palette[1],
        '1001-1500': color_palette[2],
        '1501-2000': color_palette[3],
        '2001-2500': color_palette[4],
        '2501-3000': color_palette[5],
        '3001-3500': color_palette[6],
        '3501-4000': color_palette[7],
        '4001-4500': color_palette[8],
        f'4501-{max_frequency}': above_22000_color
    }
    country_data['color'] = bins.map(lambda bin_label: bin_color_map[bin_label])
    country_data['hover_text'] = country_data.apply(
        lambda row: f"Country: {row['Country']}<br>Freq: {row['Frequency']}<br>{row['Percentage']:.2f}%", axis=1)
    top_5_countries = country_data.nlargest(5, 'Frequency')
    fig = px.choropleth(
        country_data,
        locations='Country',
        locationmode='country names',
        color='Frequency',
        hover_name='Country',
        hover_data={'Frequency': True, 'Percentage': True, 'color': False},
        title='World Map with Country Frequencies',
        color_continuous_scale=color_palette + [above_22000_color]
    )
    for _, row in top_5_countries.iterrows():
        fig.add_trace(
            go.Scattergeo(
                locationmode='country names',
                locations=[row['Country']],
                text=row['Country'],
                mode='text',
                showlegend=False,
                textfont=dict(
                    size=16,
                    color='black'
                )
            )
        )
    fig.update_layout(
        width=1600,
        height=1200,
        coloraxis_colorbar=dict(
            title='Frequency',
            tickvals=[0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, max_frequency],
            ticktext=['0', '2000', '4000', '6000', '8000', '10000', '12000', '14000', '16000', '18000', '20000', '22000']
        )
    )
    fig.write_image("world_map_frequencies.pdf")
    fig.show()



def main():
    file_path = './retractions35215.csv'
    df = load_data(file_path)
    
    columns_to_drop = ['Record ID', 'URLS', 'OriginalPaperDOI', 'RetractionDOI', 'RetractionPubMedID', 
                       'OriginalPaperPubMedID', 'Title', 'Institution', 'Notes']
    df = drop_columns(df, columns_to_drop)
    df = drop_null_paywalled(df)
    df = convert_dates(df)
    df = create_time_difference(df)
    df = create_retraction_to_citation_ratio(df)
    df = create_citation_to_retraction_ratio(df)
    df = apply_transform_subject(df)
    
    columns_to_analyze = ['Subject', 'Journal', 'Publisher', 'Country', 'ArticleType', 'Reason', 'Author']
    for column in columns_to_analyze:
        plot_top_10(df, column)
    
    params = {
        'Reason': {'top_n': 20, 'weighted_degree_thresh': 1, 'drop_thresh': 5},
        'Author': {'top_n': 1, 'weighted_degree_thresh': 1, 'drop_thresh': 1},
        'Subject': {'top_n': None, 'weighted_degree_thresh': 1, 'drop_thresh': 3},
        'Country': {'top_n': 100, 'weighted_degree_thresh': 1, 'drop_thresh': 2}
    }
    for column in params:
        display_network_graph(df, column, **params[column])
    
    plot_paywalled_frequency(df)
    plot_distribution_analysis(df)
    display_summary_statistics(df)
    plot_correlation_matrix(df)
    plot_time_series_analysis(df)
    
    top_n_values = {
        'Subject': 5,
        'Journal': 5,
        'Author': 5,
        'Publisher': 5,
        'Country': 5,
        'ArticleType': 5,
        'Reason': 5
    }
    for column, top_n in top_n_values.items():
        plot_time_series_by_category(df, column, top_n)
    
    df_no_paywall = df[df['Paywalled'] == 'No']
    categorical_columns_no_paywall = ['Subject', 'Journal', 'Publisher', 'Country', 'ArticleType', 'Reason']
    for category in categorical_columns_no_paywall:
        plot_retractions_and_citations(df_no_paywall, category)
    
    plot_world_map_analysis('country.csv')
    
    # Save the transformed DataFrame to a CSV file
    df.to_csv('transformed_retractions_after_EDA.csv', index=False)
    
    return df

if __name__ == '__main__':
    df_transformed = main()
    print("Column names after transformations:", df_transformed.columns.tolist())
