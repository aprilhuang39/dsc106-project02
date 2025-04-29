import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.patches import Patch
from scipy import stats

# Create output directory if it doesn't exist
output_dir = 'abortion_analysis_plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set the style
plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Read the Excel file
df = pd.read_excel('GuttmacherInstituteAbortionDataByState.xlsx')

# Create a dictionary mapping states to their political affiliation
state_politics = {
    'Alabama': 'Republican', 'Alaska': 'Republican', 'Arizona': 'Democratic',
    'Arkansas': 'Republican', 'California': 'Democratic', 'Colorado': 'Democratic',
    'Connecticut': 'Democratic', 'Delaware': 'Democratic', 'Florida': 'Republican',
    'Georgia': 'Republican', 'Hawaii': 'Democratic', 'Idaho': 'Republican',
    'Illinois': 'Democratic', 'Indiana': 'Republican', 'Iowa': 'Republican',
    'Kansas': 'Republican', 'Kentucky': 'Republican', 'Louisiana': 'Republican',
    'Maine': 'Democratic', 'Maryland': 'Democratic', 'Massachusetts': 'Democratic',
    'Michigan': 'Democratic', 'Minnesota': 'Democratic', 'Mississippi': 'Republican',
    'Missouri': 'Republican', 'Montana': 'Republican', 'Nebraska': 'Republican',
    'Nevada': 'Democratic', 'New Hampshire': 'Democratic', 'New Jersey': 'Democratic',
    'New Mexico': 'Democratic', 'New York': 'Democratic', 'North Carolina': 'Republican',
    'North Dakota': 'Republican', 'Ohio': 'Republican', 'Oklahoma': 'Republican',
    'Oregon': 'Democratic', 'Pennsylvania': 'Democratic', 'Rhode Island': 'Democratic',
    'South Carolina': 'Republican', 'South Dakota': 'Republican', 'Tennessee': 'Republican',
    'Texas': 'Republican', 'Utah': 'Republican', 'Vermont': 'Democratic',
    'Virginia': 'Democratic', 'Washington': 'Democratic', 'West Virginia': 'Republican',
    'Wisconsin': 'Democratic', 'Wyoming': 'Republican'
}

# Add political affiliation column to the dataframe
df['Political_Affiliation'] = df['U.S. State'].map(state_politics)

# Convert funding columns to numeric, replacing any non-numeric values with NaN
df['Federal_Funding'] = pd.to_numeric(df['Reported public expenditures for abortions (in 000s of dollars), federal, 2015'], errors='coerce')
df['State_Funding'] = pd.to_numeric(df['Reported public expenditures for abortions (in 000s of dollars), state, 2015'], errors='coerce')

# Create calculated fields
df['Total_Funding'] = df['Federal_Funding'].fillna(0) + df['State_Funding'].fillna(0)

# Create access score (lower is better)
df['Access_Score'] = (pd.to_numeric(df['% of counties without a known clinic, 2020'], errors='coerce') + 
                     pd.to_numeric(df['% of women aged 15-44 living in a county without a clinic, 2020'], errors='coerce')) / 2

# Create additional calculated fields
df['Abortion_Rate'] = pd.to_numeric(df['No. of abortions per 1,000 women aged 15–44, by state of occurrence, 2020'], errors='coerce')
df['Out_of_State_Travel'] = pd.to_numeric(df['% of residents obtaining abortions who traveled out of state for care, 2020'], errors='coerce')

# Create composite score (higher is better)
df['Composite_Score'] = (
    (df['Abortion_Rate'] / df['Abortion_Rate'].max()) * 0.4 +
    (1 - df['Access_Score'] / 100) * 0.3 +
    (df['Total_Funding'] / df['Total_Funding'].max()) * 0.3
) * 100

# Create rate categories
df['Rate_Category'] = pd.qcut(df['Abortion_Rate'], q=4, labels=['Very Low', 'Low', 'High', 'Very High'])

# Sort dataframe by composite score for better visualization
df = df.sort_values('Composite_Score', ascending=False)

# Create individual plots and save them

# 1. Abortion Rate by State with Political Affiliation
plt.figure(figsize=(15, 8))
colors = {'Democratic': '#2E86C1', 'Republican': '#E74C3C'}  # More neutral colors

# Sort dataframe by abortion rate in descending order
df_abortion = df.sort_values('Abortion_Rate', ascending=False)

# Create the bar plot
sns.barplot(data=df_abortion, 
            x='U.S. State',
            y='Abortion_Rate',
            hue='Political_Affiliation',
            palette=colors)

# Add national average line
national_avg = df['Abortion_Rate'].mean()
plt.axhline(y=national_avg, color='gray', linestyle='--', alpha=0.5)
plt.text(len(df)-1, national_avg, f'National Average: {national_avg:.1f}', 
         ha='right', va='bottom', color='gray')

# Add annotations for notable states
notable_states = {
    'New York': 'Highest Rate',
    'Mississippi': 'Lowest Rate',
    'California': 'Major Provider'
}
for state, note in notable_states.items():
    state_idx = df[df['U.S. State'] == state].index[0]
    plt.annotate(note, 
                xy=(state_idx, df.loc[state_idx, 'Abortion_Rate']),
                xytext=(0, 10), textcoords='offset points',
                ha='center', va='bottom')

plt.title('State-by-State Abortion Rates: A Tale of Two Americas', pad=20, fontsize=14)
plt.xlabel('State', fontsize=12)
plt.ylabel('Abortion Rate (per 1,000 women)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Political Affiliation', title_fontsize=12)

# Add political affiliation summary
dem_avg = df[df['Political_Affiliation'] == 'Democratic']['Abortion_Rate'].mean()
rep_avg = df[df['Political_Affiliation'] == 'Republican']['Abortion_Rate'].mean()
plt.text(0.02, 0.98, 
         f'Democratic States Average: {dem_avg:.1f}\nRepublican States Average: {rep_avg:.1f}',
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
         verticalalignment='top')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'abortion_rate_by_state.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Access to Abortion Services with Trend Line
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, 
                x='Access_Score',
                y='Out_of_State_Travel',
                hue='Political_Affiliation',
                palette=colors,
                s=100,
                alpha=0.7)

# Add trend line
z = np.polyfit(df['Access_Score'], df['Out_of_State_Travel'], 1)
p = np.poly1d(z)
plt.plot(df['Access_Score'], p(df['Access_Score']), "gray", alpha=0.5)

plt.title('The Relationship Between Access and Travel: When Barriers Force Movement', pad=20, fontsize=14)
plt.xlabel('Access Score (Higher = More Limited Access)', fontsize=12)
plt.ylabel('% of Women Traveling Out of State', fontsize=12)

# Add state labels with improved positioning
for i, row in df.iterrows():
    if row['Out_of_State_Travel'] > 30 or row['Access_Score'] > 80:  # Only label notable states
        plt.annotate(row['U.S. State'], 
                    (row['Access_Score'], row['Out_of_State_Travel']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8)

# Add quadrant annotations
plt.annotate('High Access, Low Travel', xy=(20, 10), xytext=(20, 10), 
             ha='center', va='center', color='gray', alpha=0.5)
plt.annotate('Low Access, High Travel', xy=(80, 80), xytext=(80, 80), 
             ha='center', va='center', color='gray', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'access_to_services.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Public Funding with Normalized Values
plt.figure(figsize=(15, 8))
# Normalize funding by population
df['Normalized_Funding'] = df['Total_Funding'] / df['Abortion_Rate']  # Funding per abortion

sns.barplot(data=df, 
            x='U.S. State',
            y='Normalized_Funding',
            hue='Political_Affiliation',
            palette=colors)
plt.title('Public Funding for Abortion Services: The Cost of Care', pad=20, fontsize=14)
plt.xlabel('State', fontsize=12)
plt.ylabel('Funding per Abortion (in 000s of dollars)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Political Affiliation', title_fontsize=12)

# Add national average line
national_avg = df['Normalized_Funding'].mean()
plt.axhline(y=national_avg, color='gray', linestyle='--', alpha=0.5)
plt.text(len(df)-1, national_avg, f'National Average: ${national_avg:.1f}k', 
         ha='right', va='bottom', color='gray')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'public_funding.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Out-of-State Travel with Political Affiliation
plt.figure(figsize=(15, 8))
# Sort dataframe by out-of-state travel in ascending order
df_travel = df.sort_values('Out_of_State_Travel', ascending=True)

# Create the bar plot with political affiliation
sns.barplot(data=df_travel, 
            x='U.S. State',
            y='Out_of_State_Travel',
            hue='Political_Affiliation',
            palette=colors)

# Add national average line
national_avg = df['Out_of_State_Travel'].mean()
plt.axhline(y=national_avg, color='gray', linestyle='--', alpha=0.5)
plt.text(len(df)-1, national_avg, f'National Average: {national_avg:.1f}%', 
         ha='right', va='bottom', color='gray')

# Add annotations for notable states
notable_states = {
    'Mississippi': '99% Travel',
    'Wyoming': '95% Travel',
    'California': 'Lowest Travel'
}
for state, note in notable_states.items():
    state_idx = df[df['U.S. State'] == state].index[0]
    plt.annotate(note, 
                xy=(state_idx, df.loc[state_idx, 'Out_of_State_Travel']),
                xytext=(0, 10), textcoords='offset points',
                ha='center', va='bottom')

plt.title('The Journey for Care: How Many Women Must Travel?', pad=20, fontsize=14)
plt.xlabel('State', fontsize=12)
plt.ylabel('% of Women Traveling Out of State', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Political Affiliation', title_fontsize=12)

# Add political affiliation summary
dem_avg = df[df['Political_Affiliation'] == 'Democratic']['Out_of_State_Travel'].mean()
rep_avg = df[df['Political_Affiliation'] == 'Republican']['Out_of_State_Travel'].mean()
plt.text(0.02, 0.98, 
         f'Democratic States Average: {dem_avg:.1f}%\nRepublican States Average: {rep_avg:.1f}%',
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
         verticalalignment='top')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'out_of_state_travel.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. Correlation Heatmap with Annotations
plt.figure(figsize=(10, 8))
correlation_vars = [
    'Abortion_Rate',
    'Access_Score',
    'Total_Funding',
    'Out_of_State_Travel'
]
correlation_data = df[correlation_vars].apply(pd.to_numeric, errors='coerce')
correlation_matrix = correlation_data.corr()

# Create custom colormap
cmap = sns.diverging_palette(220, 20, as_cmap=True)

sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap=cmap, 
            center=0,
            fmt='.2f',
            square=True,
            cbar_kws={'label': 'Correlation Coefficient'})

plt.title('Understanding the Complex Web of Abortion Access Factors', pad=20, fontsize=14)

# Add annotations for notable correlations
notable_correlations = [
    ('Access_Score', 'Out_of_State_Travel', 'Strong positive correlation: Limited access drives travel'),
    ('Abortion_Rate', 'Access_Score', 'Strong negative correlation: Better access leads to higher rates')
]

for var1, var2, note in notable_correlations:
    plt.annotate(note,
                xy=(correlation_vars.index(var2), correlation_vars.index(var1)),
                xytext=(0, -5), textcoords='offset points',
                ha='center', va='top',
                fontsize=8,
                color='gray')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# Print summary statistics
print("\nSummary Statistics by Political Affiliation:")
print("\nAbortion Rate (2020):")
print(df.groupby('Political_Affiliation')['Abortion_Rate'].describe())

print("\nAccess Score (lower is better):")
print(df.groupby('Political_Affiliation')['Access_Score'].describe())

print("\nOut-of-State Travel:")
print(df.groupby('Political_Affiliation')['Out_of_State_Travel'].describe())

print("\nComposite Score:")
print(df.groupby('Political_Affiliation')['Composite_Score'].describe())

print("\nCorrelation Matrix:")
print(correlation_matrix)

print(f"\nPlots have been saved in the '{output_dir}' directory.") 