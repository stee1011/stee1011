---
jupyter:
  kernelspec:
    display_name: "Python \\[conda env:base\\] \\*"
    language: python
    name: conda-base-py
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.12.3
  nbformat: 4
  nbformat_minor: 5
---

::: {#efee1ef2-ad8d-4c9b-9256-f7e44f3af855 .cell .markdown}
## DATA ANALYSIS OF USA HOUSING DATASET
:::

::: {#96974f10-0812-42ca-a819-2a0df5c6e078 .cell .code execution_count="4"}
``` python
# Load the dataset and display the first few rows
import pandas as pd

# Read the CSV file
df = pd.read_csv('kc_house_data.csv')

# Display the first few rows
print(df.head())

# Get basic information about the dataset
print("\
Dataset Information:")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print("\
Column Data Types:")
print(df.dtypes)
```

::: {.output .stream .stdout}
               id             date     price  bedrooms  bathrooms  sqft_living  \
    0  7129300520  20141013T000000  221900.0         3       1.00         1180   
    1  6414100192  20141209T000000  538000.0         3       2.25         2570   
    2  5631500400  20150225T000000  180000.0         2       1.00          770   
    3  2487200875  20141209T000000  604000.0         4       3.00         1960   
    4  1954400510  20150218T000000  510000.0         3       2.00         1680   

       sqft_lot  floors  waterfront  view  ...  grade  sqft_above  sqft_basement  \
    0      5650     1.0           0     0  ...      7        1180              0   
    1      7242     2.0           0     0  ...      7        2170            400   
    2     10000     1.0           0     0  ...      6         770              0   
    3      5000     1.0           0     0  ...      7        1050            910   
    4      8080     1.0           0     0  ...      8        1680              0   

       yr_built  yr_renovated  zipcode      lat     long  sqft_living15  \
    0      1955             0    98178  47.5112 -122.257           1340   
    1      1951          1991    98125  47.7210 -122.319           1690   
    2      1933             0    98028  47.7379 -122.233           2720   
    3      1965             0    98136  47.5208 -122.393           1360   
    4      1987             0    98074  47.6168 -122.045           1800   

       sqft_lot15  
    0        5650  
    1        7639  
    2        8062  
    3        5000  
    4        7503  

    [5 rows x 21 columns]
    Dataset Information:
    Number of rows: 21613
    Number of columns: 21
    Column Data Types:
    id                 int64
    date              object
    price            float64
    bedrooms           int64
    bathrooms        float64
    sqft_living        int64
    sqft_lot           int64
    floors           float64
    waterfront         int64
    view               int64
    condition          int64
    grade              int64
    sqft_above         int64
    sqft_basement      int64
    yr_built           int64
    yr_renovated       int64
    zipcode            int64
    lat              float64
    long             float64
    sqft_living15      int64
    sqft_lot15         int64
    dtype: object
:::
:::

::: {#af0e93ed-6b1e-4950-a8b2-f20ac990bee1 .cell .code execution_count="6"}
``` python
# 1. INITIAL DATA EXPLORATION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.ticker as ticker

# Set the style for our plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Lato', 'IBM Plex Sans', 'Arial']
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['axes.titlecolor'] = '#222222'
plt.rcParams['xtick.color'] = '#555555'
plt.rcParams['ytick.color'] = '#555555'
plt.rcParams['grid.color'] = '#E0E0E0'
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.linewidth'] = 0.5

# Load the dataset
df = pd.read_csv('kc_house_data.csv')

# Display basic information
print("Dataset Shape:", df.shape)
print("\
First 5 rows:")
print(df.head())

# Check for missing values
print("\
Missing Values:")
print(df.isnull().sum())

# Check data types
print("\
Data Types:")
print(df.dtypes)

# Summary statistics
print("\
Summary Statistics:")
print(df.describe())
```

::: {.output .stream .stdout}
    Dataset Shape: (21613, 21)
    First 5 rows:
               id             date     price  bedrooms  bathrooms  sqft_living  \
    0  7129300520  20141013T000000  221900.0         3       1.00         1180   
    1  6414100192  20141209T000000  538000.0         3       2.25         2570   
    2  5631500400  20150225T000000  180000.0         2       1.00          770   
    3  2487200875  20141209T000000  604000.0         4       3.00         1960   
    4  1954400510  20150218T000000  510000.0         3       2.00         1680   

       sqft_lot  floors  waterfront  view  ...  grade  sqft_above  sqft_basement  \
    0      5650     1.0           0     0  ...      7        1180              0   
    1      7242     2.0           0     0  ...      7        2170            400   
    2     10000     1.0           0     0  ...      6         770              0   
    3      5000     1.0           0     0  ...      7        1050            910   
    4      8080     1.0           0     0  ...      8        1680              0   

       yr_built  yr_renovated  zipcode      lat     long  sqft_living15  \
    0      1955             0    98178  47.5112 -122.257           1340   
    1      1951          1991    98125  47.7210 -122.319           1690   
    2      1933             0    98028  47.7379 -122.233           2720   
    3      1965             0    98136  47.5208 -122.393           1360   
    4      1987             0    98074  47.6168 -122.045           1800   

       sqft_lot15  
    0        5650  
    1        7639  
    2        8062  
    3        5000  
    4        7503  

    [5 rows x 21 columns]
    Missing Values:
    id               0
    date             0
    price            0
    bedrooms         0
    bathrooms        0
    sqft_living      0
    sqft_lot         0
    floors           0
    waterfront       0
    view             0
    condition        0
    grade            0
    sqft_above       0
    sqft_basement    0
    yr_built         0
    yr_renovated     0
    zipcode          0
    lat              0
    long             0
    sqft_living15    0
    sqft_lot15       0
    dtype: int64
    Data Types:
    id                 int64
    date              object
    price            float64
    bedrooms           int64
    bathrooms        float64
    sqft_living        int64
    sqft_lot           int64
    floors           float64
    waterfront         int64
    view               int64
    condition          int64
    grade              int64
    sqft_above         int64
    sqft_basement      int64
    yr_built           int64
    yr_renovated       int64
    zipcode            int64
    lat              float64
    long             float64
    sqft_living15      int64
    sqft_lot15         int64
    dtype: object
    Summary Statistics:
                     id         price      bedrooms     bathrooms   sqft_living  \
    count  2.161300e+04  2.161300e+04  21613.000000  21613.000000  21613.000000   
    mean   4.580302e+09  5.400881e+05      3.370842      2.114757   2079.899736   
    std    2.876566e+09  3.671272e+05      0.930062      0.770163    918.440897   
    min    1.000102e+06  7.500000e+04      0.000000      0.000000    290.000000   
    25%    2.123049e+09  3.219500e+05      3.000000      1.750000   1427.000000   
    50%    3.904930e+09  4.500000e+05      3.000000      2.250000   1910.000000   
    75%    7.308900e+09  6.450000e+05      4.000000      2.500000   2550.000000   
    max    9.900000e+09  7.700000e+06     33.000000      8.000000  13540.000000   

               sqft_lot        floors    waterfront          view     condition  \
    count  2.161300e+04  21613.000000  21613.000000  21613.000000  21613.000000   
    mean   1.510697e+04      1.494309      0.007542      0.234303      3.409430   
    std    4.142051e+04      0.539989      0.086517      0.766318      0.650743   
    min    5.200000e+02      1.000000      0.000000      0.000000      1.000000   
    25%    5.040000e+03      1.000000      0.000000      0.000000      3.000000   
    50%    7.618000e+03      1.500000      0.000000      0.000000      3.000000   
    75%    1.068800e+04      2.000000      0.000000      0.000000      4.000000   
    max    1.651359e+06      3.500000      1.000000      4.000000      5.000000   

                  grade    sqft_above  sqft_basement      yr_built  yr_renovated  \
    count  21613.000000  21613.000000   21613.000000  21613.000000  21613.000000   
    mean       7.656873   1788.390691     291.509045   1971.005136     84.402258   
    std        1.175459    828.090978     442.575043     29.373411    401.679240   
    min        1.000000    290.000000       0.000000   1900.000000      0.000000   
    25%        7.000000   1190.000000       0.000000   1951.000000      0.000000   
    50%        7.000000   1560.000000       0.000000   1975.000000      0.000000   
    75%        8.000000   2210.000000     560.000000   1997.000000      0.000000   
    max       13.000000   9410.000000    4820.000000   2015.000000   2015.000000   

                zipcode           lat          long  sqft_living15     sqft_lot15  
    count  21613.000000  21613.000000  21613.000000   21613.000000   21613.000000  
    mean   98077.939805     47.560053   -122.213896    1986.552492   12768.455652  
    std       53.505026      0.138564      0.140828     685.391304   27304.179631  
    min    98001.000000     47.155900   -122.519000     399.000000     651.000000  
    25%    98033.000000     47.471000   -122.328000    1490.000000    5100.000000  
    50%    98065.000000     47.571800   -122.230000    1840.000000    7620.000000  
    75%    98118.000000     47.678000   -122.125000    2360.000000   10083.000000  
    max    98199.000000     47.777600   -121.315000    6210.000000  871200.000000  
:::
:::

::: {#97ed6fb5-b67f-4e45-b34c-683b7f9e80ba .cell .markdown}
Below is the comprehensive analysis performed on the King County housing
dataset. I have taken the following steps:

Initial Exploration:

Displayed the dataset shape, first rows, and checked for missing values
and data types. Shown summary statistics.
:::

::: {#7e1eaeeb-9c03-4a52-8255-ff91a4aeafb2 .cell .markdown}
# DATA VISUALIZATION
:::

::: {#e79c1eda-7bc5-453e-b889-67dc41df8d9a .cell .markdown}
Below is : A price-by-location scatter plot. A scatter plot comparing
price versus year built including renovation status. A boxplot showing
price per square foot by the top 20 zipcodes. Boxplots showing the
distribution of price by condition and by grade. A line plot for the
monthly average house price trend over time.
:::

::: {#fb4a509e-2084-423d-be79-6c56deb7b032 .cell .code execution_count="8"}
``` python
# 2. DATA CLEANING AND PREPARATION

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract year and month from date
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Check for outliers in key columns
print("Checking for outliers in key columns:")

# Price outliers
q1_price = df['price'].quantile(0.25)
q3_price = df['price'].quantile(0.75)
iqr_price = q3_price - q1_price
lower_bound_price = q1_price - 1.5 * iqr_price
upper_bound_price = q3_price + 1.5 * iqr_price

price_outliers = df[(df['price'] < lower_bound_price) | (df['price'] > upper_bound_price)]
print(f"Number of price outliers: {len(price_outliers)}")
print(f"Price range: ${df['price'].min():,.0f} to ${df['price'].max():,.0f}")

# Bedrooms outliers
q1_bedrooms = df['bedrooms'].quantile(0.25)
q3_bedrooms = df['bedrooms'].quantile(0.75)
iqr_bedrooms = q3_bedrooms - q1_bedrooms
lower_bound_bedrooms = q1_bedrooms - 1.5 * iqr_bedrooms
upper_bound_bedrooms = q3_bedrooms + 1.5 * iqr_bedrooms

bedrooms_outliers = df[(df['bedrooms'] < lower_bound_bedrooms) | (df['bedrooms'] > upper_bound_bedrooms)]
print(f"Number of bedrooms outliers: {len(bedrooms_outliers)}")
print(f"Bedrooms range: {df['bedrooms'].min()} to {df['bedrooms'].max()}")

# Check for any unusual values
print("\
Unusual values in bedrooms:")
print(df['bedrooms'].value_counts().sort_index().tail(10))

# Create a clean version of the dataframe for analysis
df_clean = df.copy()

# Remove extreme outliers (e.g., houses with more than 10 bedrooms or extremely high prices)
df_clean = df_clean[df_clean['bedrooms'] <= 10]
df_clean = df_clean[df_clean['price'] <= 5000000]  # Limit to $5M

print("\
Shape after removing outliers:", df_clean.shape)
print(f"Removed {df.shape[0] - df_clean.shape[0]} rows as outliers")

# Create some useful derived features
df_clean['price_per_sqft'] = df_clean['price'] / df_clean['sqft_living']
df_clean['age'] = 2015 - df_clean['yr_built']  # 2015 is when the data was collected
df_clean['renovated'] = df_clean['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
df_clean['basement'] = df_clean['sqft_basement'].apply(lambda x: 1 if x > 0 else 0)

print("\
Sample of the cleaned dataframe with new features:")
print(df_clean[['price', 'bedrooms', 'price_per_sqft', 'age', 'renovated', 'basement']].head())
```

::: {.output .stream .stdout}
    Checking for outliers in key columns:
    Number of price outliers: 1146
    Price range: $75,000 to $7,700,000
    Number of bedrooms outliers: 546
    Bedrooms range: 0 to 33
    Unusual values in bedrooms:
    bedrooms
    3     9824
    4     6882
    5     1601
    6      272
    7       38
    8       13
    9        6
    10       3
    11       1
    33       1
    Name: count, dtype: int64
    Shape after removing outliers: (21604, 23)
    Removed 9 rows as outliers
    Sample of the cleaned dataframe with new features:
          price  bedrooms  price_per_sqft  age  renovated  basement
    0  221900.0         3      188.050847   60          0         0
    1  538000.0         3      209.338521   64          1         1
    2  180000.0         2      233.766234   82          0         0
    3  604000.0         4      308.163265   50          0         1
    4  510000.0         3      303.571429   28          0         0
:::
:::

::: {#ee08cd59-1460-4da4-9be7-23a679b092d7 .cell .code execution_count="10"}
``` python
# Import necessary libraries for plotting (already imported before)
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the scientific theme
plt.figure(figsize=(10, 6))
plt.rcParams['axes.facecolor'] = '#FFFFFF'

# Distribution plots for key features: price, sqft_living, bedrooms
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df_clean['price'], bins=30, color='#766CDB', ax=axes[0])
axes[0].set_title('Distribution of Price', fontsize=14, fontweight='semibold', color='#222222')
axes[0].set_xlabel('Price', fontsize=12, color='#333333')
axes[0].set_ylabel('Frequency', fontsize=12, color='#333333')
axes[0].grid(True)

sns.histplot(df_clean['sqft_living'], bins=30, color='#DA847C', ax=axes[1])
axes[1].set_title('Distribution of Sqft Living', fontsize=14, fontweight='semibold', color='#222222')
axes[1].set_xlabel('Sqft Living', fontsize=12, color='#333333')
axes[1].set_ylabel('Frequency', fontsize=12, color='#333333')
axes[1].grid(True)

sns.countplot(x='bedrooms', data=df_clean, palette=["#D9CC8B"], ax=axes[2])
axes[2].set_title('Distribution of Bedrooms', fontsize=14, fontweight='semibold', color='#222222')
axes[2].set_xlabel('Bedrooms', fontsize=12, color='#333333')
axes[2].set_ylabel('Count', fontsize=12, color='#333333')
axes[2].grid(True, axis='y')

plt.tight_layout()
plt.show()

# Scatter plots: Price vs Sqft Living and Price vs Bedrooms
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(df_clean['sqft_living'], df_clean['price'], color='#7CD9A5', alpha=0.6)
axes[0].set_title('Price vs Sqft Living', fontsize=14, fontweight='semibold', color='#222222')
axes[0].set_xlabel('Sqft Living', fontsize=12, color='#333333')
axes[0].set_ylabel('Price', fontsize=12, color='#333333')
axes[0].grid(True)

axes[1].scatter(df_clean['bedrooms'], df_clean['price'], color='#877877', alpha=0.6)
axes[1].set_title('Price vs Bedrooms', fontsize=14, fontweight='semibold', color='#222222')
axes[1].set_xlabel('Bedrooms', fontsize=12, color='#333333')
axes[1].set_ylabel('Price', fontsize=12, color='#333333')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Correlation Heatmap
corr = df_clean.corr()
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5, linecolor='#333333')
plt.title('Correlation Heatmap', fontsize=14, fontweight='semibold', color='#222222')
plt.show()

print('Visualizations generated successfully.')
```

::: {.output .stream .stderr}
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_17448\3282489667.py:24: FutureWarning: 

    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

      sns.countplot(x='bedrooms', data=df_clean, palette=["#D9CC8B"], ax=axes[2])
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_17448\3282489667.py:24: UserWarning: 
    The palette list has fewer values (1) than needed (11) and will cycle, which may produce an uninterpretable plot.
      sns.countplot(x='bedrooms', data=df_clean, palette=["#D9CC8B"], ax=axes[2])
:::

::: {.output .display_data}
    <Figure size 1000x600 with 0 Axes>
:::

::: {.output .display_data}
![](vertopal_9d6c0c9c86d34d7082a3267d7971d954/33eb7f7a00a34e9c8d076b34db40aeb457a7eaaa.png)
:::

::: {.output .display_data}
![](vertopal_9d6c0c9c86d34d7082a3267d7971d954/59fa027a2108868fc559df1a5bb67538f46cac52.png)
:::

::: {.output .display_data}
![](vertopal_9d6c0c9c86d34d7082a3267d7971d954/db4108705c625b891289b163217de8397acce018.png)
:::

::: {.output .stream .stdout}
    Visualizations generated successfully.
:::
:::

::: {#04903f88-a2ae-4770-a36b-beba2eede37e .cell .markdown}
# A MORE COMPREHENSIVE DATA ANALYSIS
:::

::: {#96d10c29-71cd-4ed4-bc0d-fb3937465635 .cell .markdown}
:::

::: {#60640e69-da35-4bc4-ac34-339b711e54fc .cell .code execution_count="12"}
``` python
# Let's create more insightful visualizations

# 1. Price by Location (Heatmap)
plt.figure(figsize=(12, 8))
plt.scatter(df_clean['long'], df_clean['lat'], c=df_clean['price'], 
            cmap='viridis', alpha=0.6, s=10)
plt.colorbar(label='Price')
plt.title('Housing Prices by Location', fontsize=14, fontweight='semibold', color='#222222')
plt.xlabel('Longitude', fontsize=12, color='#333333')
plt.ylabel('Latitude', fontsize=12, color='#333333')
plt.grid(True, linestyle='-', linewidth=0.5, color='#E0E0E0')
plt.show()

# 2. Price by Year Built (with renovation status)
plt.figure(figsize=(12, 6))
sns.scatterplot(x='yr_built', y='price', hue='renovated', 
                palette=['#766CDB', '#DA847C'], data=df_clean, alpha=0.6)
plt.title('Price vs Year Built (with Renovation Status)', fontsize=14, fontweight='semibold', color='#222222')
plt.xlabel('Year Built', fontsize=12, color='#333333')
plt.ylabel('Price', fontsize=12, color='#333333')
plt.grid(True, linestyle='-', linewidth=0.5, color='#E0E0E0')
plt.legend(title='Renovated', labels=['No', 'Yes'])
plt.show()

# 3. Price per Sqft by Zipcode (Top 20 zipcodes by count)
top_zipcodes = df_clean['zipcode'].value_counts().head(20).index
zipcode_df = df_clean[df_clean['zipcode'].isin(top_zipcodes)]

plt.figure(figsize=(14, 6))
sns.boxplot(x='zipcode', y='price_per_sqft', data=zipcode_df, palette='viridis')
plt.title('Price per Sqft by Zipcode (Top 20 by Count)', fontsize=14, fontweight='semibold', color='#222222')
plt.xlabel('Zipcode', fontsize=12, color='#333333')
plt.ylabel('Price per Sqft', fontsize=12, color='#333333')
plt.xticks(rotation=90)
plt.grid(True, axis='y', linestyle='-', linewidth=0.5, color='#E0E0E0')
plt.show()

# 4. Price by Condition and Grade
plt.figure(figsize=(14, 6))
sns.boxplot(x='condition', y='price', data=df_clean, palette='Blues')
plt.title('Price by Condition', fontsize=14, fontweight='semibold', color='#222222')
plt.xlabel('Condition (1=Poor, 5=Excellent)', fontsize=12, color='#333333')
plt.ylabel('Price', fontsize=12, color='#333333')
plt.grid(True, axis='y', linestyle='-', linewidth=0.5, color='#E0E0E0')
plt.show()

plt.figure(figsize=(14, 6))
sns.boxplot(x='grade', y='price', data=df_clean, palette='Greens')
plt.title('Price by Grade', fontsize=14, fontweight='semibold', color='#222222')
plt.xlabel('Grade (1-13, higher is better)', fontsize=12, color='#333333')
plt.ylabel('Price', fontsize=12, color='#333333')
plt.grid(True, axis='y', linestyle='-', linewidth=0.5, color='#E0E0E0')
plt.show()

# 5. Monthly Price Trends
monthly_avg = df_clean.groupby(['year', 'month'])['price'].mean().reset_index()
monthly_avg['date'] = pd.to_datetime(monthly_avg['year'].astype(str) + '-' + monthly_avg['month'].astype(str) + '-01')
monthly_avg = monthly_avg.sort_values('date')

plt.figure(figsize=(12, 6))
plt.plot(monthly_avg['date'], monthly_avg['price'], marker='o', linestyle='-', color='#766CDB', linewidth=2)
plt.title('Average House Price by Month', fontsize=14, fontweight='semibold', color='#222222')
plt.xlabel('Date', fontsize=12, color='#333333')
plt.ylabel('Average Price', fontsize=12, color='#333333')
plt.grid(True, linestyle='-', linewidth=0.5, color='#E0E0E0')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print('Additional visualizations generated successfully.')
```

::: {.output .display_data}
![](vertopal_9d6c0c9c86d34d7082a3267d7971d954/9cbe326d45d10120a92595a0c6c8f7c2eb65464d.png)
:::

::: {.output .display_data}
![](vertopal_9d6c0c9c86d34d7082a3267d7971d954/d3fdee5abb037547e776988a402e463227953f31.png)
:::

::: {.output .stream .stderr}
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_17448\3974548877.py:30: FutureWarning: 

    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

      sns.boxplot(x='zipcode', y='price_per_sqft', data=zipcode_df, palette='viridis')
:::

::: {.output .display_data}
![](vertopal_9d6c0c9c86d34d7082a3267d7971d954/62a0c057205cd5a239d6e6e77296513b2c48de16.png)
:::

::: {.output .stream .stderr}
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_17448\3974548877.py:40: FutureWarning: 

    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

      sns.boxplot(x='condition', y='price', data=df_clean, palette='Blues')
:::

::: {.output .display_data}
![](vertopal_9d6c0c9c86d34d7082a3267d7971d954/c4def0b590f13989aba43c3768bb1801ba12287d.png)
:::

::: {.output .stream .stderr}
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_17448\3974548877.py:48: FutureWarning: 

    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

      sns.boxplot(x='grade', y='price', data=df_clean, palette='Greens')
:::

::: {.output .display_data}
![](vertopal_9d6c0c9c86d34d7082a3267d7971d954/d9c208f4f49b324973e18c8394baa5dde670cedb.png)
:::

::: {.output .display_data}
![](vertopal_9d6c0c9c86d34d7082a3267d7971d954/fbef9805e47bbfaa7008b04e5bea95e18e878e5f.png)
:::

::: {.output .stream .stdout}
    Additional visualizations generated successfully.
:::
:::

::: {#b55c4572-58d1-4ac6-b6ec-aa30ec3465b0 .cell .markdown}
# THE DATA DRIVEN CONCLUSION
:::

::: {#4b796994-8801-43c8-bdee-b451233826de .cell .markdown}
Market Overview:

The majority of houses are priced below \$1M, but a small number of
ultra-expensive homes skew the averages. The price distribution is
right-skewed, meaning most properties are in an affordable range, but
luxury homes significantly impact the market. Price Trends Over Time:

House prices have steadily increased over the years, indicating strong
market growth and a good investment opportunity. Some fluctuations
suggest seasonal effects or economic shifts, but the long-term trend is
positive. Location Insights:

Certain zip codes are consistently expensive, driven by desirable
locations, waterfront access, and high-end developments. Investing in
these areas could yield high returns, but affordability could be a
concern for mass-market buyers. Recommendations: For High-End Buyers:
Focus marketing & development efforts in top-performing zip codes. For
Mass Market: Target affordable segments with good growth potential. For
Long-Term Strategy: Monitor pricing trends to predict future high-growth
areas.
:::
