# Phase 1 Project Description

## Project Overview

Slicken Company is expanding into the aviation industry to diversify its portfolio. This involves purchasing and operating airplanes for commercial and private enterprises. It is therefore imperative that an analysis is undertaken to assess the risks associated with aviation to support business in making informed data-driven decisions to help avert high liability, operational losses, or reputational damage that may occur.

## Business Context

Slicken Company currently lacks information about the potential risks associated with different types of aircraft. To support this new venture, a project has been undertaken, tasked with analyzing historical aircraft safety data to identify which aircraft types present the lowest operational risk. The findings will be translated into actionable insights to guide the head of the new aviation division in making informed, data-driven decisions about which aircraft to purchase for both commercial and private operations.


## The Data

In the `data` folder is a [dataset](https://www.kaggle.com/datasets/khsamaha/aviation-accident-database-synopses) from the National Transportation Safety Board that includes aviation accident data from 1962 to 2023 about civil aviation accidents and selected incidents in the United States and international waters.

## Goals
1.Determining which aircraft are the lowest risk for the company to start this new business endeavor.
2.Support strategic aircraft acquisition through data-driven insights.
labels, titles).

## Deliverables
There are three deliverables for this project:

* A **non-technical presentation**
* A **Jupyter Notebook**
* A **GitHub repository**
* An **Interactive Dashboard**

# Importing libraries

import pandas as pd # for data manipulation and analysis
import numpy as np   # for numerical operations
import matplotlib.pyplot as plt # for plotting and visualizations
import seaborn as sns # for statistical data visualization

%matplotlib inline

#Importing/Converting Aviation Data into a DataFrame for Exploratory Data Analysis 
df = pd.read_csv('data/Aviation_Data.csv', low_memory=False)
df.head()

# Exploratory Data Analysis with pandas

## Initial data inspection
df.info() #to get an overview of the dataset

"""The dataset has 31 columns with 90348 rows. The data types is a mixture of objects(26) and floats(5). The dataset has alot of missing values,only Investigation.Type column has no missing value. This means there is need to conduct data cleaning first before EDA"""

#Generating summary statistics for numeric columns.
df.describe()

## Data Cleaning
Data cleaning is the process of detecting and correcting (or removing) inaccurate, incomplete, inconsistent, or irrelevant data from a dataset to improve its quality and reliability for analysis.

#Total number of duplicated rows
df.duplicated().sum()

#View duplicated rows
df[df.duplicated()]

#Dropping duplicates and assigning a new variable df_clean
df_clean = df.drop_duplicates()
df_clean.shape

#Removing white spaces from df_clean
string_cols = df_clean.select_dtypes(include='object').columns # Identify string (object) columns
df_clean = df[string_cols].apply(lambda x: x.str.strip())

#checking for missing values
df_clean.isnull().sum()

#drop columns that will not help in data analysis
drop_cols = [
    'Event.Id',                 
    'Accident.Number',          
    'Airport.Code',             
    'Aircraft.Category',        
    'FAR.Description',          
    'Schedule',                 
    'Air.carrier',             
    'Report.Status',           
    'Publication.Date'          
]

#Dropping the columns from df_clean dataframe
df_clean = df.drop(columns=drop_cols)

#viewing the remaining columns
df_clean.columns

#Confirming the columns are dropped
df_clean.shape
df_clean.info()

#Converting column latitude and Longitude from type objects to type floats
df_clean[['Latitude', 'Longitude']] = df_clean[['Latitude', 'Longitude']].app

df_clean['Weather.Condition'].unique()
#Standardize case for the weather column
df_clean['Weather.Condition'] = df_clean['Weather.Condition'].str.upper().str.strip()

#Fixing the data type for Event.Date column
df_clean['Event.Date']=pd.to_datetime(df_clean['Event.Date'], errors ='coerce')

df_clean['Broad.phase.of.flight'].unique()

#Normalize Case & Strip Whitespace
df_clean.loc[:, 'Make'] = df_clean['Make'].str.upper().str.strip()

#filter out rows with fewer than 3 characters or all uppercase names with spaces (like people's names)
df_clean = df_clean[df_clean['Make'].str.len() > 2]

# Data Exploration(EDA)
Now that we have cleaned the data. We now carry out Exploratory Data Analysis which involves examining and visualizing a dataset to understand its structure, patterns, trends, and relationships before formal modeling or hypothesis testing.

#Extract year
df_clean['Year']=df_clean['Event.Date'].dt.year
df_clean.head()

#Converting Year column to type string
df_clean['Year'] = df_clean['Year'].astype(float).astype(str)

#Top aircrafts with most accidents
df_clean['Make'].value_counts()

#Get the top 10 most common aircraft makes
top_makes = df_clean['Make'].value_counts().head(10)

#Plot
top_makes.plot(kind='bar', figsize=(10, 6), color='skyblue')
plt.title('Top 10 Aircraft Makes in Accident Records')
plt.xlabel('Aircraft Make')
plt.ylabel('Number of Records')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
<img width="730" height="442" alt="image" src="https://github.com/user-attachments/assets/2a6ea160-85da-432f-8674-ea197b58e3a5" />


#Fatal Injuries by engine type
FatalitiesByEngineType = df_clean.groupby('Engine.Type')['Total.Fatal.Injuries'].sum().sort_values(ascending=False)
FatalitiesByEngineType


#Plotting a graph of FatalitiesByEngineType vs Total Fatal Injuries
FatalitiesByEngineType.plot(kind='bar', figsize=(10, 6), color='green')

plt.title('Total Fatalities by Engine Type')
plt.xlabel('Phase of Flight')
plt.ylabel('Total Fatal Injuries')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

<img width="767" height="447" alt="image" src="https://github.com/user-attachments/assets/ae560e85-bdc6-4e7c-991b-e2f3498b47dc" />


#Plotting a graph of Fatalities By Phase Of Flight vs Total Fatal Injuries
FatalitiesByPhaseOfFlight= df_clean.groupby('Broad.phase.of.flight')['Total.Fatal.Injuries'].sum().sort_values(ascending=False)
FatalitiesByPhaseOfFlight

#Fatalities by phase of flight
FatalitiesByPhaseOfFlight.plot(kind='bar', figsize=(10, 6), color='salmon')

plt.title('Total Fatalities by Phase of Flight')
plt.xlabel('Phase of Flight')
plt.ylabel('Total Fatal Injuries')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

<img width="752" height="442" alt="image" src="https://github.com/user-attachments/assets/a54dbdd1-f064-4cd2-94e6-0436252dbb85" />


#Yearly accident trend
df_clean['Year'].value_counts().sort_index().plot(kind='line', figsize=(10, 5))

plt.xlabel('Year')                     
plt.ylabel('Number of Accidents')      
plt.title('Aircraft Accidents Over Time') 
plt.grid(True)                       

plt.tight_layout()
plt.show()


<img width="757" height="368" alt="image" src="https://github.com/user-attachments/assets/949c5864-a436-4a5a-b5d8-05c810be2f89" />

#Export file as CSV for visualization
df_clean.to_csv('Aviation_Data_Analysis.csv', index=False)

# Summary
1.Amateur-built aircrafts had low accident rates and low fatalities/injuries compared to non-amateur-built aircrafts.
2.Most accidents occurred when the weather was VMC (minimum airspeed at which a twin-engine aircraft can still be controlled safely if one engine fails, while the other is operating at full power).
3.Turbo Fan engine had the highest number of recorded survivors at 221,048.
4.Aircrafts of Cessna, Boeing, and Piper makes recorded the highest rates of fatalities respectively.
5.Personal flights indicated the highest number of fatalities compared to business, executive and corporate flights.
<img width="1337" height="705" alt="image" src="https://github.com/user-attachments/assets/7ca0fa71-2187-408b-8517-4e4726663953" />

# Recommendations
1.The company should consider buying amateur-built aircrafts since it had low accident rates compared to non-amateur-built aircrafts.
2.The company should consider investing in Turbo Fan engine aircraft since it had the highest number of recorded survivors.
3.The company should avoid/minimize the purchase of Aircrafts of Cessna, Boeing, and Piper makes since these have recorded the highest rates of fatalities respectively.
4.The company should consider investing more in business, executive and corporate flights and less in personal flights since it indicated the highest number of fatalities.



