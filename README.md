# Matplotlib_hw
## Observations and Insights 



# Dependencies and Setup
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import numpy as np
from scipy.stats import sem
from sklearn.linear_model import LinearRegression

# Study data files
mouse_metadata_path = "data/Mouse_metadata.csv"
study_results_path = "data/Study_results.csv"

# Read the mouse data and the study results
mouse_metadata = pd.read_csv(mouse_metadata_path)
study_results = pd.read_csv(study_results_path)

# Combine the data into a single dataset
combined_data_df = pd.merge(mouse_metadata, study_results, how='outer', on='Mouse ID')

# Display the data table for preview
combined_data_df.head(10)

# Checking the number of mice.
combined_data_df.count()

# Getting the duplicate mice by ID number that shows up for Mouse ID and Timepoint. 

mice_and_timepoint_duplicates = combined_data_df[combined_data_df.duplicated(['Mouse ID', 'Timepoint'])]
mice_and_timepoint_duplicates


# Optional: Get all the data for the duplicate mouse ID. 

mice_duplicates = combined_data_df[combined_data_df.duplicated(['Mouse ID'])]
mice_duplicates

# Create a clean DataFrame by dropping the duplicate mouse by its ID.
combined_data_df_clean = combined_data_df.drop_duplicates(subset=['Mouse ID'])
combined_data_df_clean

combined_data_df_clean_renamed = combined_data_df_clean.rename(columns={"Age_months": "Age", "Weight (g)": "Weight", "Tumor Volume (mm3)": "Tumor Volume"})            
combined_data_df_clean_renamed.head()

# Checking the number of mice in the clean DataFrame.
print(combined_data_df_clean_renamed["Mouse ID"].count())

## Summary Statistics

# Generate a summary statistics table of mean, median, variance, standard deviation, and SEM of the tumor volume for each regimen
# Use groupby and summary statistical methods to calculate the following properties of each drug regimen: 
# mean, median, variance, standard deviation, and SEM of the tumor volume. 
mean_tumor_volume = combined_data_df.groupby('Drug Regimen')['Tumor Volume (mm3)'].mean()
median_tumor_volume = combined_data_df.groupby('Drug Regimen')['Tumor Volume (mm3)'].median()
var_tumor_volume = combined_data_df.groupby('Drug Regimen')['Tumor Volume (mm3)'].var()
dev_tumor_volume = combined_data_df.groupby('Drug Regimen')['Tumor Volume (mm3)'].std()
sem_tumor_volume = combined_data_df.groupby('Drug Regimen')['Tumor Volume (mm3)'].sem()

# Assemble the resulting series into a single summary dataframe.
summary_one_df = pd.DataFrame({"Mean": mean_tumor_volume, "Median": median_tumor_volume, "Variance": var_tumor_volume, 
                               "Standard Deviation": dev_tumor_volume, "Sem": sem_tumor_volume})
summary_one_df



summary_one_df[["Sem"]]

# Generate a summary statistics table of mean, median, variance, standard deviation, and SEM of the tumor volume for each regimen
# Using the aggregation method, produce the same summary statistics in a single line

single_line_df = combined_data_df.groupby('Drug Regimen').agg(['mean', 'median', 'var', 'std', 'sem'])["Tumor Volume (mm3)"]
single_line_df 

## Bar and Pie Charts

# Generate a bar plot showing the total number of measurements taken on each drug regimen using pandas.
regimen_groups = combined_data_df.groupby('Drug Regimen')
measurement_count = regimen_groups['Mouse ID'].count()
regimen_count_bar = measurement_count.plot(kind="bar", figsize=(11,4), title="Total Number of Measurements by Drug Regimen")
regimen_count_bar.set_xlabel("Drug Regimen")
regimen_count_bar.set_ylabel("Number of Measurements Taken")

# Generate a bar plot showing the total number of measurements taken on each drug regimen using pyplot.
regimen_list = combined_data_df['Drug Regimen'].values
x_axis_values = np.unique(regimen_list)
measurement_list = combined_data_df.groupby(["Drug Regimen"])["Mouse ID"]
y_axis_values = measurement_list.count()
plt.figure(figsize=(11,4))
plt.bar(x_axis_values, y_axis_values, color='r', alpha=0.5, align="center")
plt.title("Total Number of Measurements by Drug Regimen")
plt.xlabel("Drug Regimen")
plt.ylabel("Number of Measurements Taken")
plt.show()  

# Generate a pie plot showing the distribution of female versus male mice using pandas
sex_groups_df = pd.DataFrame(combined_data_df.groupby(['Sex']).count())
sex_groups_df.head()

sex_pie_one = sex_groups_df.plot(kind="pie", y='Mouse ID', title=("Gender Count"))

sex_count = combined_data_df['Sex'].value_counts()
sex_count

# Generate a pie plot showing the distribution of female versus male mice using pyplot
# Pie chart, where the slices will be ordered and plotted counter-clockwise:


labels = 'male', 'female'
sex_count = [958, 935]
explode = (0.1, 0)

fig1, ax1 = plt.subplots()
ax1.pie(sex_count, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=160)
ax1.axis('equal') 

plt.show()

## Quartiles, Outliers and Boxplots

# Calculate the final tumor volume of each mouse across four of the treatment regimens: 
# Capomulin, Ramicane, Infubinol, and Ceftamin

# Start by getting the last (greatest) timepoint for each mouse
timepoint_max = combined_data_df.groupby('Mouse ID').max()['Timepoint']

# Merge this group df with the original dataframe to get the tumor volume at the last timepoint
timepoint_max_df = pd.DataFrame(timepoint_max)
merged_timepoint_df = pd.merge(timepoint_max_df, combined_data_df, how='outer', on=('Mouse ID','Timepoint'))
merged_timepoint_df.head()

# Put treatments into a list for for loop (and later for plot labels)
list_of_regimens = ['Capomulin', 'Ramicane', 'Infubinol', 'Ceftamin']

# Create empty list to fill with tumor vol data (for plotting)
cap_list = []
ram_list = []
infu_list = []
ceft_list = []
    
# Calculate the IQR and quantitatively determine if there are any potential outliers. 

for index, row in combined_data_df.iterrows():
    if row['Drug Regimen'] == 'Capomulin':
        cap_list.append(row["Tumor Volume (mm3)"])
    if row['Drug Regimen'] == 'Ramicane':
        ram_list.append(row["Tumor Volume (mm3)"])
    if row['Drug Regimen'] == 'Infubinol':
        infu_list.append(row["Tumor Volume (mm3)"])
    if row['Drug Regimen'] == 'Ceftamin':
        ceft_list.append(row["Tumor Volume (mm3)"])
    
cap_iqr = st.iqr(cap_list)
ram_iqr = st.iqr(ram_list)
infu_iqr = st.iqr(infu_list)
ceft_iqr = st.iqr(ceft_list)

cap_outliers = [x for x in cap_list if x < np.quantile(cap_list,.25) - 1.5 * cap_iqr or 
                        x > np.quantile(cap_list,.75) + 1.5 * cap_iqr]
ram_outliers = [x for x in ram_list if x < np.quantile(ram_list,.25) - 1.5 * ram_iqr or 
                        x > np.quantile(ram_list,.75) + 1.5 * ram_iqr]
infu_outliers = [x for x in infu_list if x < np.quantile(infu_list,.25) - 1.5 * infu_iqr or 
                        x > np.quantile(infu_list,.75) + 1.5 * infu_iqr]
ceft_outliers = [x for x in ceft_list if x < np.quantile(ceft_list,.25) - 1.5 * ceft_iqr or 
                        x > np.quantile(ceft_list,.75) + 1.5 * ceft_iqr]

print(cap_outliers)
print(ram_outliers)
print(infu_outliers)
print(ceft_outliers)

# Generate a box plot of the final tumor volume of each mouse across four regimens of interest
plt.boxplot([cap_list, ram_list, infu_list, ceft_list], flierprops=dict(marker='o',markerfacecolor='r'));
plt.xticks([1,2,3,4],["Capomulin", "Ramicane", "Infubinol", "Ceftamin"]);
plt.ylabel("Tumor Volume (mm3)");

## Line and Scatter Plots

# Generate a line plot of tumor volume vs. time point for a mouse treated with Capomulin
cap_df = combined_data_df[(combined_data_df['Drug Regimen']=='Capomulin') & (combined_data_df['Mouse ID']=='s185')]
plt.plot(cap_df["Timepoint"],cap_df["Tumor Volume (mm3)"])
plt.xlabel("Timepoint")
plt.ylabel("Tumor Volume (mm3)")

# Generate a scatter plot of average tumor volume vs. mouse weight for the Capomulin regimen
cap_df = combined_data_df[combined_data_df['Drug Regimen']=='Capomulin']
plt.scatter(cap_df.groupby("Mouse ID")["Tumor Volume (mm3)"].mean(), cap_df.groupby("Mouse ID")["Weight (g)"].mean())
plt.xlabel("Tumor Volume (mm3)");
plt.ylabel("Weight (g)");

## Correlation and Regression

# Calculate the correlation coefficient and linear regression model 
# for mouse weight and average tumor volume for the Capomulin regimen
x = cap_df.groupby("Mouse ID")["Tumor Volume (mm3)"].mean().to_numpy().reshape(-1,1)
y = cap_df.groupby("Mouse ID")["Weight (g)"].mean().tolist()
reg = LinearRegression().fit(x, y)
r2 = reg.score(x, y)
print("Correlation Coefficient = " + str(np.sqrt(r2)))
