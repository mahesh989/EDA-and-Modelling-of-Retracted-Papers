import pandas as pd
import re
df = pd.read_csv('./retraction_before_modelling.csv')


df.shape
df.head()
df.columns

df.head()
column_to_delete = ['Author','OriginalPaperDate','RetractionNature','RetractionDate']
df = df.drop(column_to_delete, axis=1)

df.duplicated().sum()
df.drop_duplicates(inplace=True)

df.head()

# Define the columns to analyze
columns_to_analyze = ['Subject', 'Journal', 'Publisher', 'Country', 'ArticleType', 'Reason']

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


# Function to replace rows with the most frequent individual entry
def replace_with_most_frequent(column, occurrences):
    for i, cell in df[column].dropna().iteritems():
        entries = re.split(r'[;,]', cell)
        if len(entries) > 1:
            most_frequent_entry = max(entries, key=lambda x: occurrences.get(x.strip(), 0))
            df.loc[i, column] = most_frequent_entry.strip()  # Use .loc to avoid SettingWithCopyWarning


# Apply the steps for each column
for column in columns_to_analyze:
    occurrences = count_occurrences(column)
    replace_with_most_frequent(column, occurrences)
    
df['Retraction_to_Mean_Days_Ratio'] = df['Retraction_to_Citation_Ratio'] / df['TimeDifference_Days']


df.columns
df['Reason'] = df['Reason'].str.replace('+','',regex=False)

df.head()

df.to_csv('ready_for_one_hot_encoding.csv', index=False)