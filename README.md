## Grading Criteria

1. **Uploaded the URL of your GitHub repository including all the completed notebooks and Python files** (1 pt)
2. **Uploaded your completed presentation in PDF format** (1 pt)
3. **Completed the required Executive Summary slide** (1 pt)
4. **Completed the required Introduction slide** (1 pt)
5. **Completed the required data collection and data wrangling methodology related slides** (1 pt)
6. **Completed the required EDA and interactive visual analytics methodology related slides** (3 pts)
7. **Completed the required predictive analysis methodology related slides** (1 pt)
8. **Completed the required EDA with visualization results slides** (6 pts)
9. **Completed the required EDA with SQL results slides** (10 pts)
10. **Completed the required interactive map with Folium results slides** (3 pts)
11. **Completed the required Plotly Dash dashboard results slides** (3 pts)
12. **Completed the required predictive analysis (classification) results slides** (6 pts)
13. **Completed the required Conclusion slide** (1 pt)
14. **Applied your creativity to improve the presentation beyond the template** (1 pt)
15. **Displayed any innovative insights** (1 pt)

### You will not be judged on:
- Your English language, including spelling or grammatical mistakes
- The content of any text or image(s) or where a link is hyperlinked to

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import folium
from folium.plugins import MarkerCluster

# Part 1: Data Visualization Tasks

# TASK 1.1: Line Plot using Pandas
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020],
    'Sales': [20000, 22000, 25000, 23000, 24000, 21000]
}
df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
plt.plot(df['Year'], df['Sales'], marker='o')
plt.title('Automobile Sales from Year to Year')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.grid(True)
plt.savefig('Line_plot_1.png')
plt.show()

# TASK 1.2: Line Plot for Different Vehicle Types
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020],
    'Sedan': [15000, 16000, 17000, 16500, 17000, 15000],
    'SUV': [5000, 6000, 8000, 6500, 7000, 6000]
}
df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
plt.plot(df['Year'], df['Sedan'], marker='o', label='Sedan')
plt.plot(df['Year'], df['SUV'], marker='o', label='SUV')
plt.title('Vehicle Sales by Type')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.savefig('Line_plot_2.png')
plt.show()

# TASK 1.3: Seaborn Visualization
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020],
    'Type': ['Sedan', 'SUV', 'Sedan', 'SUV', 'Sedan', 'SUV'],
    'Sales': [15000, 6000, 17000, 8000, 17000, 7000],
    'Period': ['Non-Recession', 'Non-Recession', 'Recession', 'Recession', 'Non-Recession', 'Recession']
}
df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
sns.barplot(x='Year', y='Sales', hue='Type', data=df)
plt.title('Sales Trend per Vehicle Type')
plt.savefig('Bar_Chart.png')
plt.show()

# TASK 1.4: Subplots for GDP Variations
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020],
    'GDP_Recession': [1.5, 1.6, 1.4, 1.2, 1.3, 1.1],
    'GDP_Non_Recession': [2.5, 2.6, 2.4, 2.2, 2.3, 2.1]
}
df = pd.DataFrame(data)

fig, ax = plt.subplots(2, 1, figsize=(10, 12))

ax[0].plot(df['Year'], df['GDP_Recession'], marker='o')
ax[0].set_title('GDP During Recession')
ax[0].set_xlabel('Year')
ax[0].set_ylabel('GDP')

ax[1].plot(df['Year'], df['GDP_Non_Recession'], marker='o')
ax[1].set_title('GDP During Non-Recession')
ax[1].set_xlabel('Year')
ax[1].set_ylabel('GDP')

plt.tight_layout()
plt.savefig('Subplot.png')
plt.show()

# TASK 1.5: Bubble Plot
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020],
    'Sales': [20000, 22000, 25000, 23000, 24000, 21000],
    'Seasonality': [1.2, 1.5, 1.7, 1.3, 1.6, 1.4]
}
df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
plt.scatter(df['Year'], df['Sales'], s=df['Seasonality']*1000, alpha=0.5)
plt.title('Impact of Seasonality on Automobile Sales')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.grid(True)
plt.savefig('Bubble.png')
plt.show()

# TASK 1.6: Scatter Plot with Matplotlib
data = {
    'Average_Price': [20000, 22000, 25000, 23000, 24000, 21000],
    'Sales_Volume': [500, 600, 700, 550, 650, 580]
}
df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
plt.scatter(df['Average_Price'], df['Sales_Volume'])
plt.title('Correlation between Average Vehicle Price and Sales Volume during Recessions')
plt.xlabel('Average Vehicle Price')
plt.ylabel('Sales Volume')
plt.grid(True)
plt.savefig('Scatter.png')
plt.show()

# TASK 1.7: Pie Chart for Advertising Expenditure
data = {
    'Period': ['Recession', 'Non-Recession'],
    'Expenditure': [30000, 50000]
}
df = pd.DataFrame(data)

plt.figure(figsize=(8, 8))
plt.pie(df['Expenditure'], labels=df['Period'], autopct='%1.1f%%')
plt.title('Advertising Expenditure of XYZAutomotives')
plt.savefig('Pie_1.png')
plt.show()

# TASK 1.8: Pie Chart for Advertisement Expenditure by Vehicle Type
data = {
    'Vehicle_Type': ['Sedan', 'SUV', 'Truck'],
    'Expenditure': [20000, 15000, 10000]
}
df = pd.DataFrame(data)

plt.figure(figsize=(8, 8))
plt.pie(df['Expenditure'], labels=df['Vehicle_Type'], autopct='%1.1f%%')
plt.title('Advertisement Expenditure by Vehicle Type during Recession')
plt.savefig('Pie_2.png')
plt.show()

# TASK 1.9: Line Plot for Unemployment Rate and Vehicle Sales
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020],
    'Unemployment_Rate': [5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
    'Sedan_Sales': [15000, 16000, 17000, 16500, 17000, 15000],
    'SUV_Sales': [5000, 6000, 8000, 6500, 7000, 6000]
}
df = pd.DataFrame(data)

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(df['Year'], df['Unemployment_Rate'], color='red', marker='o', label='Unemployment Rate')
ax1.set_xlabel('Year')
ax1.set_ylabel('Unemployment Rate', color='red')

ax2 = ax1.twinx()
ax2.plot(df['Year'], df['Sedan_Sales'], color='blue', marker='o', label='Sedan Sales')
ax2.plot(df['Year'], df['SUV_Sales'], color='green', marker='o', label='SUV Sales')
ax2.set_ylabel('Vehicle Sales', color='blue')

fig.tight_layout()
plt.title('Unemployment Rate and Vehicle Sales during Recession Period')
fig.savefig('Line_plot_3.png')
plt.show()

# Part 2: Dash Application Tasks

# Sample DataFrame for Dash app
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020],
    'Sedan_Sales': [15000, 16000, 17000, 16500, 17000, 15000],
    'SUV_Sales': [5000, 6000, 8000, 6500, 7000, 6000]
}
df = pd.DataFrame(data)

recession_data = {
    'Year': [2017, 2018, 2019, 2020],
    'Sedan_Sales': [17000, 16500, 17000, 15000],
    'SUV_Sales': [8000, 6500, 7000, 6000]
}
recession_df = pd.DataFrame(recession_data)

yearly_data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020],
    'Sedan_Sales': [15000, 16000, 17000, 16500, 17000, 15000],
    'SUV_Sales': [5000, 6000, 8000, 6500, 7000, 6000]
}
yearly_df = pd.DataFrame(yearly_data)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Automobile Sales Dashboard"),
    dcc.Dropdown(
        id='vehicle-type-dropdown',
        options=[
            {'label': 'Sedan', 'value': 'Sedan_Sales'},
            {'label': 'SUV', 'value': 'SUV_Sales'}
        ],
        placeholder="Select a vehicle type"
    ),
    html.Div(id='output-container', className='output-container'),
    dcc.Graph(id='line-plot'),
    dcc.Graph(
        id='recession-report',
        figure=px.line(recession_df, x='Year', y=['Sedan_Sales', 'SUV_Sales'], title='Recession Report Statistics')
    ),
    dcc.Graph(
        id='yearly-report',
        figure=px.line(yearly_df, x='Year', y=['Sedan_Sales', 'SUV_Sales'], title='Yearly Report Statistics')
    )
])

@app.callback(
    Output('line-plot', 'figure'),
    Input('vehicle-type-dropdown', 'value')
)
def update_graph(selected_vehicle):
    if selected_vehicle is None:
        return {}
    
    fig = px.line(df, x='Year', y=selected_vehicle, title=f'Sales Trend for {selected_vehicle.split("_")[0]}')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)


