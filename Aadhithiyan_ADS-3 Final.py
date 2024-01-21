import pandas as pd ## import Pandas for data manipulation 
import numpy as np
import matplotlib.pyplot as plt #Matplotlib for plotting
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures # for data analysis and modeling, including various machine learning algorithms for classification, regression, clustering
from sklearn.linear_model import LinearRegression
import seaborn as sns # Imports Seaborn, which is built on top of Matplotlib

# Read the data
df = pd.read_csv("API_19_DS2_en_csv_v2_6300757.csv", skiprows=3)

"""
    Perform KMeans clustering on two indicators and plot the clustered data.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - indicator1 (str): The name of the first indicator.
    - indicator2 (str): The name of the second indicator.
    - year_column (str): The column representing the year.
    - k_optimal1 (int): The optimal number of clusters for the first indicator.
    - k_optimal2 (int): The optimal number of clusters for the second indicator.
    """

# Filter data for the specified indicators and year (1990)
data1 = df[df['Indicator Name'] == 'CO2 emissions (kt)'][['Country Name', '1990']].rename(columns={'1990': 'CO2 emissions (kt)'})
data2 = df[df['Indicator Name'] == 'Agricultural land (sq. km)'][['Country Name', '1990']].rename(columns={'1990': 'Agricultural land (sq. km)'})

merged_data = pd.merge(data1, data2, on='Country Name', how='outer').reset_index(drop=True)
merged_data.dropna(inplace=True)

# Filter data for the specified indicators and year (2020)
data3 = df[df['Indicator Name'] == 'CO2 emissions (kt)'][['Country Name', '2020']].rename(columns={'2020': 'CO2 emissions (kt)'})
data4 = df[df['Indicator Name'] == 'Agricultural land (sq. km)'][['Country Name', '2020']].rename(columns={'2020': 'Agricultural land (sq. km)'})

merged_data2 = pd.merge(data3, data4, on='Country Name', how='outer').reset_index(drop=True)
merged_data2.dropna(inplace=True)

# Create elbow plots for merged_data and merged_data2 in a single plot
X_merged_data = merged_data[['CO2 emissions (kt)', 'Agricultural land (sq. km)']].dropna()
X_merged_data2 = merged_data2[['CO2 emissions (kt)', 'Agricultural land (sq. km)']].dropna()

sse_merged_data = []
sse_merged_data2 = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)

    kmeans.fit(X_merged_data)
    sse_merged_data.append(kmeans.inertia_)

    kmeans.fit(X_merged_data2)
    sse_merged_data2.append(kmeans.inertia_)

# Plotting the Elbow plots in a single plot
plt.figure(figsize=(12, 6))

plt.plot(range(1, 11), sse_merged_data, marker='o', label='merged_data')
plt.plot(range(1, 11), sse_merged_data2, marker='o', label='merged_data2')

plt.title('Elbow Plots for merged_data and merged_data2')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Distances')
plt.legend()

plt.show()

# Optimal number of clusters obtained from the elbow plots
k_optimal_merged_data = 3
k_optimal_merged_data2 = 4

# Perform KMeans clustering
kmeans_merged_data = KMeans(n_clusters=k_optimal_merged_data, random_state=42)
kmeans_merged_data.fit(X_merged_data)

kmeans_merged_data2 = KMeans(n_clusters=k_optimal_merged_data2, random_state=42)
kmeans_merged_data2.fit(X_merged_data2)

# Add cluster labels and center points to the dataframes
merged_data['Cluster'] = kmeans_merged_data.labels_
merged_data['Center_X'] = merged_data.groupby('Cluster')['CO2 emissions (kt)'].transform('mean')
merged_data['Center_Y'] = merged_data.groupby('Cluster')['Agricultural land (sq. km)'].transform('mean')

merged_data2['Cluster'] = kmeans_merged_data2.labels_
merged_data2['Center_X'] = merged_data2.groupby('Cluster')['CO2 emissions (kt)'].transform('mean')
merged_data2['Center_Y'] = merged_data2.groupby('Cluster')['Agricultural land (sq. km)'].transform('mean')

# Plotting the clustering results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x='CO2 emissions (kt)', y='Agricultural land (sq. km)', hue='Cluster', data=merged_data, palette='viridis')
plt.scatter(merged_data['Center_X'].unique(), merged_data['Center_Y'].unique(), marker='x', color='black', s=50, label='Cluster Center')
plt.title('Clustered Data for merged_data')

plt.subplot(1, 2, 2)
sns.scatterplot(x='CO2 emissions (kt)', y='Agricultural land (sq. km)', hue='Cluster', data=merged_data2, palette='viridis')
plt.scatter(merged_data2['Center_X'].unique(), merged_data2['Center_Y'].unique(), marker='x', color='black', s=50, label='Cluster Center')
plt.title('Clustered Data for merged_data2')

plt.tight_layout()
plt.show()



# Set a custom color palette for the plot
sns.set_palette("husl")

# Read the data
df = pd.read_csv('API_19_DS2_en_csv_v2_6300757.csv', skiprows=3)

# Select three countries and the indicator
selected_countries = ['Brazil', 'India']
indicator_name = 'CO2 emissions (kt)'

# Filter the data
data_selected = df[(df['Country Name'].isin(selected_countries)) & (df['Indicator Name'] == indicator_name)].reset_index(drop=True)

# Melt the DataFrame
data_forecast = data_selected.melt(id_vars=['Country Name', 'Indicator Name'], var_name='Year', value_name='Value')

# Filter out non-numeric values in the 'Year' column
data_forecast = data_forecast[data_forecast['Year'].str.isnumeric()]

# Convert 'Year' to integers
data_forecast['Year'] = data_forecast['Year'].astype(int)

# Handle NaN values by filling with the mean value
data_forecast['Value'].fillna(data_forecast['Value'].mean(), inplace=True)

# Filter data for the years between 1990 and 2025
data_forecast = data_forecast[(data_forecast['Year'] >= 1990) & (data_forecast['Year'] <= 2025)]

# Create a dictionary to store predictions for each country
predictions = {}

"""
    Perform polynomial regression, forecast for 2025, and plot for selected countries.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - selected_countries (list): List of countries for forecasting.
    - indicator_name (str): The name of the indicator for forecasting.
    """

# Extend the range of years to include 2025
all_years_extended = list(range(1990, 2026))

# Create individual plots for each country with a grid and unique style
for country in selected_countries:
    plt.figure(figsize=(7, 4))
    
    # Use line plot for actual data
    plt.plot(data_forecast[data_forecast['Country Name'] == country]['Year'], 
         data_forecast[data_forecast['Country Name'] == country]['Value'], 
         marker='o', linestyle='-', label=f'Actual Data', color='blue')
    
    # Prepare data for the current country
    country_data = data_forecast[data_forecast['Country Name'] == country]
    country_data = country_data.sort_values(by='Year')  # Sort by 'Year'
    
    X_country = country_data[['Year']]
    y_country = country_data['Value']
    
    # Fit polynomial regression model with degree 4
    degree = 4
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X_country)
    
    model = LinearRegression()
    model.fit(X_poly, y_country)
    
    # Predict values for all years (1990 to 2025)
    X_pred = poly_features.transform(pd.DataFrame(all_years_extended, columns=['Year']))
    forecast_values = model.predict(X_pred)
    
    # Store the predictions for the current country
    predictions[country] = forecast_values
    
    # Plot the fitted curve
    plt.plot(all_years_extended, forecast_values, label=f'Fitted Curve', linestyle='-', color='red')
    
    # Plot forecast for 2025
    prediction_2025 = forecast_values[-1]
    plt.scatter(2025, prediction_2025, marker='o', s=100, label=f'Prediction for 2025: {prediction_2025:.2f}', color='brown')
    
    plt.title(f'{indicator_name} Forecast for {country}', fontsize=12)
    plt.xlabel('Year', fontsize=10)
    plt.ylabel('Kilotonns', fontsize=10)
    
    # Set x-axis limits and ticks
    plt.xlim(1990, 2030)
    plt.xticks(range(1990, 2030, 5))  # Adjust the step as needed
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.legend(fontsize=7)
    filename = f"{indicator_name}_Forecast_{country.replace(' ', '_')}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.show()  # Add this line to display each plot separately
