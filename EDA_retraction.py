import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import re

# Load the data
file_path = './retractions35215.csv'
df = pd.read_csv(file_path)
data = df

'''
################################################################
######### Data Cleaning and Transformation ########################
###############################################################
'''
# Drop specified columns
columns_to_drop = ['Record ID', 'URLS', 'OriginalPaperDOI', 'RetractionDOI', 'RetractionPubMedID', 
                   'OriginalPaperPubMedID', 'Title', 'Institution', 'Notes']
df.drop(columns=columns_to_drop, inplace=True)

# Drop rows with null values in the 'Paywalled' column
df.dropna(subset=['Paywalled'], inplace=True)

# Overview after dropping columns and null values
print("Shape after Dropping Columns and Null Values:", df.shape)
print("Duplicated Rows after Dropping:", df.duplicated().sum())
print("Null Values after Dropping:\n", df.isnull().sum())

# Check for duplicate entries and display the first 5
duplicate_entries = df[df.duplicated(keep=False)]
print("First 5 Duplicate Entries:\n", duplicate_entries.head(5))

# Convert date columns to datetime
df['OriginalPaperDate'] = pd.to_datetime(df['OriginalPaperDate'], dayfirst=True, errors='coerce')
df['RetractionDate'] = pd.to_datetime(df['RetractionDate'], dayfirst=True, errors='coerce')

# Create new columns for the time difference between OriginalPaperDate and RetractionDate
df['TimeDifference_Days'] = (df['RetractionDate'] - df['OriginalPaperDate']).dt.days

# Create a new column for retraction to citation ratio
df['Retraction_to_Citation_Ratio'] = df.apply(lambda row: row['TimeDifference_Days'] / row['CitationCount'] if row['CitationCount'] != 0 else float('inf'), axis=1)

# Create a new column for citation to retraction ratio
df['Citation_to_Retraction_Ratio'] = df.apply(lambda row: row['CitationCount'] / row['TimeDifference_Days'] if row['TimeDifference_Days'] != 0 else float('inf'), axis=1)

# Transform the Subject column
def transform_subject(subject):
    codes = re.findall(r'\((.*?)\)', subject)
    return ', '.join(sorted(set(codes)))

df['Subject'] = df['Subject'].apply(transform_subject)
'''
################################################################
######### Occurrences Analysis #################################
###############################################################
'''
# Define the columns to analyze
columns_to_analyze = ['Subject', 'Journal', 'Publisher', 'Country', 'ArticleType', 'Reason','Author']

# Function to count occurrences of each individual entry within the cells of a column
def count_occurrences(column):
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

# Function to plot the top 10 most frequently occurring entries for a column
def plot_top_10(column):
    occurrences = count_occurrences(column)
    sorted_occurrences = dict(sorted(occurrences.items(), key=lambda item: item[1], reverse=True)[:10])
    plt.figure(figsize=(10, 5))
    plt.bar(sorted_occurrences.keys(), sorted_occurrences.values())
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Top 10 Most Frequently Occurring Entries in {column}')
    plt.xticks(rotation=45, ha='right')
    plt.show()

# Plot for each specified column
for column in columns_to_analyze:
    plot_top_10(column)
'''
###############################################################
######### Network Graph Analysis ##############################
###############################################################
'''
def display_network_graph(column_name, top_n=None, weighted_degree_thresh=1, drop_thresh=5):
    # Step 1: Data Preparation
    all_vals = ";".join(list(data[column_name].dropna())).split(";")
    all_vals = [vv.strip() for vv in all_vals if vv not in ["", "Unknown", "unavailable", "Unavailable", "No affiliation available"]]
    all_vals_series = pd.Series(all_vals)
    top_rank = all_vals_series.value_counts().sort_values(ascending=False)
    
    if top_n is not None:
        more_than_1 = list(top_rank.head(top_n).index)
    else:
        more_than_1 = list(top_rank.index)
    
    print(f"\nTop {top_n if top_n is not None else 'all'} values in {column_name}:")
    print(top_rank.head(top_n if top_n is not None else len(top_rank)))

    if len(more_than_1) == 0:
        print(f"No entries for {column_name}. Skipping...")
        return

    # Step 2: Compute the correlation matrix
    aa_cor_mat = pd.DataFrame(np.zeros((len(more_than_1), len(more_than_1))), columns=more_than_1, index=more_than_1)

    column_lists = list(data[column_name].dropna().str.split(";"))
    for cl in column_lists:
        cl = [item.strip() for item in cl]  # Ensure all items are stripped
        for ii in range(len(cl)):
            if cl[ii] not in more_than_1:
                continue
            for jj in range(ii + 1, len(cl)):
                if cl[jj] not in more_than_1:
                    continue
                aa_cor_mat.loc[cl[ii], cl[jj]] += 1
                aa_cor_mat.loc[cl[jj], cl[ii]] += 1
    
    print(f"\nInitial correlation matrix for {column_name} (first 5 rows and columns):")
    print(aa_cor_mat.iloc[:5, :5])

    # Optional: Reduce the correlation matrix
    aa_cor_mat_reduced = aa_cor_mat.copy()
    for au in aa_cor_mat.columns:
        if aa_cor_mat[au].sum() < drop_thresh:
            aa_cor_mat_reduced = aa_cor_mat_reduced.drop(columns=[au])
            aa_cor_mat_reduced = aa_cor_mat_reduced.drop(index=[au])
    
    print(f"\nReduced correlation matrix for {column_name} after applying drop_thresh of {drop_thresh} (first 5 rows and columns):")
    print(aa_cor_mat_reduced.iloc[:5, :5])

    if aa_cor_mat_reduced.shape[0] == 0 or aa_cor_mat_reduced.shape[1] == 0:
        print(f"Empty reduced correlation matrix for {column_name}. Skipping...")
        return

    # Step 3: Network Graph using reduced correlation matrix
    G = nx.from_pandas_adjacency(aa_cor_mat_reduced)

    # Filter nodes with degrees below a certain threshold
    high_degree_nodes = [node for node, degree in G.degree(weight="weight") if degree > weighted_degree_thresh]
    G = G.subgraph(high_degree_nodes)
    
    print(f"\nNodes in the final graph for {column_name}: {len(G.nodes())}")
    print(list(G.nodes())[:5])  # Print first 5 nodes as a sample
    
    print(f"\nEdges in the final graph for {column_name}: {len(G.edges())}")
    print(list(G.edges(data=True))[:5])  # Print first 5 edges as a sample

    if len(high_degree_nodes) == 0:
        print(f"No high degree nodes for {column_name} after threshold filtering. Skipping...")
        return

    # Calculate degree and weighted degree
    degrees = dict(G.degree())
    weighted_degrees = dict(G.degree(weight='weight'))
    
    print(f"\nDegree of nodes in the final graph for {column_name} (first 5):")
    print({k: degrees[k] for k in list(degrees)[:5]})  # Print first 5 degrees as a sample
    
    print(f"\nWeighted degree of nodes in the final graph for {column_name} (first 5):")
    print({k: weighted_degrees[k] for k in list(weighted_degrees)[:5]})  # Print first 5 weighted degrees as a sample

    # Find the node with the highest weighted degree
    highest_weighted_degree_node = max(weighted_degrees, key=weighted_degrees.get)
    print(f"\nNode with the highest weighted degree in the final graph for {column_name}:")
    print(f"{highest_weighted_degree_node}: {weighted_degrees[highest_weighted_degree_node]}")
    
    edge_weights = [np.log(data['weight']) for _, _, data in G.edges(data=True)]
    normalized_edge_weights = edge_weights
    degs = [d[1]**1.2 for d in G.degree(weight="weight")]

    # Plot the network graph with enhanced visualization
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=1.8)
    node_colors = range(len(G.nodes()))
    edge_colors = range(len(G.edges()))
    nx.draw(G, with_labels=True, pos=pos, font_weight='bold', edge_color=normalized_edge_weights, width=normalized_edge_weights, edge_cmap=plt.cm.Reds, edge_vmin=0, edge_vmax=4, node_size=degs, node_color=degs, cmap=plt.cm.autumn)
    plt.title(f"Network of Co-occurring {column_name}")
    plt.show()

# Define the parameters for each column
params = {
    'Reason': {'top_n': 20, 'weighted_degree_thresh': 1, 'drop_thresh': 5},
    'Author': {'top_n': 1, 'weighted_degree_thresh': 1, 'drop_thresh': 1},
    'Subject': {'top_n': None, 'weighted_degree_thresh': 1, 'drop_thresh': 3},  # Include all subjects
    'Country': {'top_n': 100, 'weighted_degree_thresh': 1, 'drop_thresh': 2}
}

# Analyze the specified columns
columns_to_analyze = ['Reason', 'Author', 'Subject', 'Country']
for column in columns_to_analyze:
    display_network_graph(column, **params[column])
    
    

'''
#################################################
######### Paywalled plot ########################
#################################################
'''
# Frequency of Paywalled values
retractions_by_paywalled = df['Paywalled'].value_counts()

# Plotting the frequency of each unique value in the Paywalled column
fig, ax = plt.subplots(figsize=(8, 6))
retractions_by_paywalled.plot(kind='bar', ax=ax, color='green', alpha=0.7)
ax.set_ylabel('Frequency')
ax.set_xlabel('Paywalled')
ax.set_title('Frequency of Unique Values in Paywalled Column')
ax.set_xticklabels(retractions_by_paywalled.index, rotation=0)
for i in range(len(retractions_by_paywalled)):
    ax.text(i, retractions_by_paywalled[i] + 5, str(retractions_by_paywalled[i]), ha='center', color='black')
plt.grid(True)
plt.show()

'''
################################################################
######### Distribution Analysis ########################
###############################################################
'''

# Replace infinite values with NaN for the purpose of exclusion in analysis
df['Retraction_to_Citation_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
df['Citation_to_Retraction_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)

# Filter out rows with NaN in the relevant columns for analysis purposes
df_filtered = df.dropna(subset=['Retraction_to_Citation_Ratio', 'Citation_to_Retraction_Ratio'])

# Distribution Analysis
plt.figure(figsize=(15, 10))
df_filtered[['CitationCount', 'TimeDifference_Days', 'Retraction_to_Citation_Ratio', 'Citation_to_Retraction_Ratio']].hist(bins=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))
for i, column in enumerate(['CitationCount', 'TimeDifference_Days', 'Retraction_to_Citation_Ratio', 'Citation_to_Retraction_Ratio'], 1):
    plt.subplot(2, 2, i)
    sns.boxplot(df_filtered[column])
    plt.title(f'Box plot of {column}')
plt.tight_layout()
plt.show()


'''
###############################################################
######### Summary Statistics ##################################
###############################################################
'''
# Summary statistics
summary_stats = df.describe()
print("Summary Statistics:\n", summary_stats)

'''
################################################################################
######### Correlation Analysis  ################################################
###############################################################################
'''
# Correlation Analysis
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


'''
########################################################
######### Time Series Analysis ########################
#######################################################
'''
# Filter the data to include only years less than 2024
df_filtered = df[df['RetractionDate'].dt.year < 2024]
df_filtered = df_filtered[df_filtered['OriginalPaperDate'].dt.year < 2023]

# Number of Retractions Over Year
df_filtered['Year_Retraction'] = df_filtered['RetractionDate'].dt.year
retractions_by_year = df_filtered['Year_Retraction'].value_counts().sort_index()

# Time Series Analysis for RetractionDate and OriginalPaperDate
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
plt.show()

# Identify the common recent years excluding the year 2023
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
plt.show()

'''
################################################################
######### Time series Analysis by categorical columns ########################
###############################################################
'''

# Filter the data to include only years less than 2024 for RetractionDate and 2023 for OriginalPaperDate
df_filtered = df[df['RetractionDate'].dt.year < 2023]


# Function to count occurrences of each individual entry within the cells of a column
def count_occurrences(column):
    occurrences = {}
    for cell in df_filtered[column].dropna():
        entries = re.split(r'[;,]', cell)
        for entry in entries:
            entry = entry.strip()
            if entry:
                if entry in occurrences:
                    occurrences[entry] += 1
                else:
                    occurrences[entry] = 1
    return occurrences

# Define the number of top values for each category
top_n_values = {
    'Subject': 5,
    'Journal': 5,
    'Author': 5,
    'Publisher': 5,
    'Country': 5,
    'ArticleType': 5,
    'Reason': 5
}

# Perform time series analysis for each categorical column
for column, top_n in top_n_values.items():
    # Count occurrences of each individual entry
    occurrences = count_occurrences(column)
    
    # Identify the top unique values by the number of retractions
    top_values = sorted(occurrences, key=occurrences.get, reverse=True)[:top_n]
    
    fig, ax_main = plt.subplots(figsize=(12, 6))
    
    for value in top_values:
        df_value_filtered = df_filtered[df_filtered[column].str.contains(value, na=False, regex=False)].copy()

        # Number of Retractions Over Year for the specific value
        df_value_filtered.loc[:, 'Year_Retraction'] = df_value_filtered['RetractionDate'].dt.year
        retractions_by_year = df_value_filtered['Year_Retraction'].value_counts().sort_index()

        ax_main.plot(retractions_by_year.index, retractions_by_year.values, marker='o', label=value)
    
    ax_main.set_title(f'Retractions Over Time for Top {top_n} {column}')
    ax_main.set_xlabel('Year')
    ax_main.set_ylabel('Count')
    ax_main.legend()
    ax_main.grid(True)

    # Create an inset plot in the middle left, except for "Author" column
    if column != 'Author':
        ax_inset = inset_axes(ax_main, width="40%", height="40%", loc="center left")
        
        # Identify the recent five years available in the data
        recent_years = retractions_by_year.index.sort_values()[-5:]
        
        for value in top_values:
            df_value_filtered = df_filtered[df_filtered[column].str.contains(value, na=False, regex=False)].copy()
            
            # Number of Retractions Over Year for the specific value
            df_value_filtered.loc[:, 'Year_Retraction'] = df_value_filtered['RetractionDate'].dt.year
            retractions_by_year = df_value_filtered['Year_Retraction'].value_counts().sort_index()

            # Filter retractions to only include recent years available in the data
            common_recent_years = recent_years.intersection(retractions_by_year.index)
            retractions_recent = retractions_by_year.loc[common_recent_years]

            ax_inset.plot(retractions_recent.index, retractions_recent.values, marker='o', label=value)
        
        # Move y-axis to the right for the inset plot
        ax_inset.yaxis.set_label_position("right")
        ax_inset.yaxis.tick_right()
        
        ax_inset.set_title('Last 5 Years')
        ax_inset.set_xlabel('Year')
        ax_inset.set_ylabel('Count')
        ax_inset.grid(True)
    
    plt.show()



'''
################################################################################
#########replace rows with the most frequent individual entry ########################
###############################################################################
'''
# # Function to replace rows with the most frequent individual entry
# def replace_with_most_frequent(column, occurrences):
#     for i, cell in df[column].dropna().iteritems():
#         entries = re.split(r'[;,]', cell)
#         if len(entries) > 1:
#             most_frequent_entry = max(entries, key=lambda x: occurrences.get(x.strip(), 0))
#             df.loc[i, column] = most_frequent_entry.strip()  # Use .loc to avoid SettingWithCopyWarning

# # Apply the steps for each column
# for column in columns_to_analyze:
#     occurrences = count_occurrences(column)
#     replace_with_most_frequent(column, occurrences)

# # Replace infinite values with NaN and then fill NaN values with the column mean
# df['Retraction_to_Citation_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
# df['Citation_to_Retraction_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
# df['Retraction_to_Citation_Ratio'].fillna(df['Retraction_to_Citation_Ratio'].mean(), inplace=True)
# df['Citation_to_Retraction_Ratio'].fillna(df['Citation_to_Retraction_Ratio'].mean(), inplace=True)




########################################
######### Analysis for Non-Paywalled ########################
#######################################

# Filter the data to include only non-paywalled retractions
df_no_paywall = df[df['Paywalled'] == 'No']

# Function to count occurrences of each individual entry within the cells of a column
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

########################################
######### Analysis for Non-Paywalled ########################
#######################################

# Filter the data to include only non-paywalled retractions
df_no_paywall = df[df['Paywalled'] == 'No']

# Function to count occurrences of each individual entry within the cells of a column
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

########################################
######### Analysis for Non-Paywalled ########################
#######################################

# Define the categorical columns to analyze
categorical_columns_no_paywall = ['Subject', 'Journal', 'Publisher', 'Country', 'ArticleType', 'Reason']

# Function to plot total number of retractions and total citations by top 5 category values
def plot_retractions_and_citations(df, category):
    # Count occurrences of each individual entry
    occurrences = count_occurrences(df, category)
    
    # Identify the top 5 unique values by the number of retractions
    top_values = sorted(occurrences, key=occurrences.get, reverse=True)[:5]
    
    # Filter the data for the top 5 values
    df_top = df[df[category].apply(lambda x: any(val in str(x) for val in top_values))]
    
    # Aggregate data for retractions
    retractions_by_category = df_top[category].apply(lambda x: [val for val in top_values if val in str(x)]).explode().value_counts()
    
    # Aggregate data for citations
    citations_by_category = {value: df_top[df_top[category].str.contains(value, na=False, regex=False)]['CitationCount'].sum() for value in top_values}
    citations_by_category = pd.Series(citations_by_category)
    
    # Create subplots
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot total number of retractions
    retractions_by_category.plot(kind='bar', ax=ax1, color='blue', alpha=0.6, position=0, width=0.4, label='Total Retractions')
    ax1.set_ylabel('Total Retractions')
    ax1.set_xlabel(category)
    ax1.set_title(f'Total Number of Retractions and Total Citations by Top 5 {category} (1990-2023, Paywalled: No)')
    
    # Create a twin Axes sharing the x-axis
    ax2 = ax1.twinx()
    
    # Plot total citations
    citations_by_category.plot(kind='bar', ax=ax2, color='red', alpha=0.6, position=1, width=0.4, label='Total Citations')
    ax2.set_ylabel('Total Citations')
    
    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.show()

# Perform the analysis and plotting for each specified category
for category in categorical_columns_no_paywall:
    plot_retractions_and_citations(df_no_paywall, category)











########################################
######### World Map Analysis ########################
#######################################

import plotly.express as px
import plotly.graph_objects as go

# Load the uploaded CSV file
file_path = 'country.csv'  # Update this path as needed
country_data = pd.read_csv(file_path)

# Calculate the total frequency and percentage
total_frequency = country_data['Frequency'].sum()
country_data['Percentage'] = (country_data['Frequency'] / total_frequency) * 100

# Define distinct color palettes for different frequency ranges
color_palette = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
    '#bcbd22', '#17becf', '#9edae5'
]
above_22000_color = '#ff1493'  # Color for frequencies above 22,000

# Create bins and assign colors based on the frequency range
max_frequency = country_data['Frequency'].max()
bins = pd.cut(country_data['Frequency'], bins=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, max_frequency],
              labels=['0-500', '501-1000', '1001-1500', '1501-2000', '2001-2500', '2501-3000', '3001-3500', '3501-4000', '4001-4500', f'4501-{max_frequency}'], include_lowest=True)

# Mapping bins to colors
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

def assign_color(bin_label):
    return bin_color_map[bin_label]

# Assign colors to each row based on its frequency bin
country_data['color'] = bins.map(assign_color)

# Create a hover text for the plot
country_data['hover_text'] = country_data.apply(
    lambda row: f"Country: {row['Country']}<br>Freq: {row['Frequency']}<br>{row['Percentage']:.2f}%", axis=1)

# Get the top 5 countries by frequency
top_5_countries = country_data.nlargest(5, 'Frequency')

# Plot the world map using Plotly
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

# Add scattergeo layer for text labels
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

# Update layout for better visibility
fig.update_layout(
    width=1600,  # Adjust the width as needed
    height=1200,  # Adjust the height as needed
    coloraxis_colorbar=dict(
        title='Frequency',
        tickvals=[0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, max_frequency],
        ticktext=['0', '2000', '4000', '6000', '8000', '10000', '12000', '14000', '16000', '18000', '20000', '22000']
    )
)

# Save the plot as a PDF
fig.write_image("world_map_frequencies.pdf")

# Show the plot
fig.show()
