import pandas as pd
import re
df = pd.read_csv('./transformed_retractions_after_EDA.csv')

df.head()
df.columns

df.head()
column_to_delete = ['Citation_to_Retraction_Ratio','Retraction_to_Citation_Ratio','RetractionNature','Author', 'TimeDifference_Days']
df = df.drop(column_to_delete, axis=1)

df.duplicated().sum()

df.drop_duplicates(inplace=True)



# Convert date columns to datetime
df['OriginalPaperDate'] = pd.to_datetime(df['OriginalPaperDate'], dayfirst=True, errors='coerce')
df['RetractionDate'] = pd.to_datetime(df['RetractionDate'], dayfirst=True, errors='coerce')

# Create new columns for the time difference between OriginalPaperDate and RetractionDate
df['TimeDifference_Days'] = (df['RetractionDate'] - df['OriginalPaperDate']).dt.days

(df['TimeDifference_Days']==1).sum()
df[df['TimeDifference_Days'] == 1]
df['TimeDifference_Days'].replace(0, 1, inplace=True)

(df['CitationCount']==1).sum()
df['CitationCount'].replace(0, 1.1, inplace=True)




# Create a new column for retraction to citation ratio
df['Retraction_to_Citation_Ratio'] = df.apply(
    lambda row: row['TimeDifference_Days'] / row['CitationCount'] if row['CitationCount'] != 0 else float('inf'), axis=1)

# Create a new column for citation to retraction ratio
df['Citation_to_Retraction_Ratio'] = df.apply(
    lambda row: row['CitationCount'] / row['TimeDifference_Days'] if row['TimeDifference_Days'] != 0 else float('inf'), axis=1)


# # Calculate the means of the respective columns
# mean_retraction_to_citation_ratio = df['Retraction_to_Citation_Ratio'].mean()
# mean_citation_to_retraction_ratio = df['Citation_to_Retraction_Ratio'].mean()

# # Identify rows where CitationCount is 0.0001
# rows_with_0001_citation_count = df['CitationCount'] == 1.1

# # Replace values in the specified columns with the mean values
# df.loc[rows_with_0001_citation_count, 'Retraction_to_Citation_Ratio'] = mean_retraction_to_citation_ratio
# df.loc[rows_with_0001_citation_count, 'Citation_to_Retraction_Ratio'] = mean_citation_to_retraction_ratio


df.isnull().sum()

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


column_to_delete = ['RetractionDate','OriginalPaperDate']
df = df.drop(column_to_delete, axis=1)


df.columns
df['Reason'] = df['Reason'].str.replace('+','',regex=False)

df.head()

df.to_csv('ready_for_modelling_1.csv', index=False)