import pandas as pd
import numpy as np
import math

import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
sns.set()
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker

# 1.1 Basic observations

# Tune the visual settings for figures in `seaborn`
sns.set_context(
    "notebook", 
    font_scale=1.5,       
    rc={ 
        "figure.figsize": (11, 8), 
        "axes.titlesize": 18 
    }
)

df = pd.read_csv("./Dataset/mlbootcamp5_train.csv")

# Check the type of the columns given in the data set

print(df.info())

# Studying the relation between the gender and the habit of smoke and drink alcohol
# Because of the height and weight we got that men is represented with the number 2 and women with the number 1

columns_to_show = ['height', 'weight', 'smoke', 'alco']
df_genders = df.groupby('gender')[columns_to_show].mean()

print(df_genders)

# Check if the age data is given in days, months or years. Its given in days

df_ages = df['age']
print(df_ages)
sns.distplot(df_ages)

plt.show()

# Separating the smokers from those that doesn't smoke so we could differentiate their mean ages

df_smokers = df[df['smoke'] == 1]['age'].median()
df_non_smokers = df[df['smoke'] == 0]['age'].median()

mean_dif = abs(((df_smokers - df_non_smokers)/365.25)*12)

print("The difference between mean values of age of those who smoke and those who doesn't smoke is %f months." % mean_dif)

# 1.2 Risk maps

# We get those individuals that are between 60 and 64 years old

age_years = (df['age']/365.25).round().astype('int')
df.insert(loc=len(df.columns), column='age_years', value=age_years)
df_older_people = df[df['age_years'].apply(lambda age: (60 <= age) and (age <= 64))]

# We select those who has cholesterol and arterial preassure lower than 120

df_op_chol_low_smoke = df_older_people[(df_older_people['cholesterol'] == 1) & (df_older_people['ap_hi'] < 120) & (df_older_people['smoke'] == 1)]
df_old_cardio = df_op_chol_low_smoke['cardio'].value_counts()
df_op_chol_low_smoke_fraction = df_old_cardio[1]/df_op_chol_low_smoke.count()

print("The proportion of older people with colesterol and low arterial preassure that has CVD is %f percent." % df_op_chol_low_smoke_fraction['cardio'])

# We got that only near to 28% of those selected had CVD. Now let's check the same thing with people with 160 to 180 of arterial preassure

df_op_chol_high_smoke = df_older_people[(df_older_people['cholesterol'] == 1) & (160 <= df_older_people['ap_hi']) & (df_older_people['ap_hi'] <= 180) & (df_older_people['smoke'] == 1)]
df_old_cardio = df_op_chol_high_smoke['cardio'].value_counts()
df_op_chol_high_smoke_fraction = df_old_cardio[1]/df_op_chol_high_smoke.count()

print("The proportion of older people with colesterol and ap between 160 and 180 that has CVD is %f percent." % df_op_chol_high_smoke_fraction['cardio'])

# Now we got that for this new group of people the percentage of them that had CVD is near 89%
# That means that those with ap between 160 and 180 are near to 3 times more likely to suffer from CVD than those with ap lower than 120

# 1.3 Analizying BMI

# Create a new feature called BMI (Body Mass Index), that is calculated from dividing the weight by the square of the height in meters. Normal values should be between 18.5 and 25

BMI = df['weight']/((df['height']/100)**2)
df.insert(loc=len(df.columns), column='BMI', value=BMI)
print(df['BMI'].describe())

# We can see that the mean BMI value doesn't fit in the range given as normal values. Our BMI mean is 27.56

df_bmi_men = df[df['gender'] == 2]['BMI'].mean()
df_bmi_women = df[df['gender'] == 1]['BMI'].mean()

print("The average BMI for men is %f and for women is %f" % (df_bmi_men, df_bmi_women))

# On average women had higher BMI than men

df_bmi_healthy = df[df['cardio'] == 0]['BMI'].mean()
df_bmi_cvd = df[df['cardio'] == 1]['BMI'].mean()

print("The average BMI for healthy people is %f and for people with cvd is %f" % (df_bmi_healthy, df_bmi_cvd))

# This says that on average healthy people had lower BMI than people with CVD

df_healthy_non_drinking_men = df_bmi_healthy = df[(df['gender'] == 2) & (df['alco'] == 0)]['BMI'].mean()
df_healthy_non_drinking_women = df_bmi_healthy = df[(df['gender'] == 1) & (df['alco'] == 0)]['BMI'].mean()

print("Healthy non-drinking men the average BMI is %f and for healthy non-drinking women the average BMI is %f" % (df_healthy_non_drinking_men, df_healthy_non_drinking_women))

# The average BMI value for healthy non-drinking men is closer to the range than for healthy non-drinking women

# 1.4 Cleaning data

# Proceed to clean the data so we keep the useful part

df_clean_data = df[(df['ap_lo'] <= df['ap_hi']) &
				   (df['height'] >= df['height'].quantile(0.025)) &
				   (df['height'] <= df['height'].quantile(0.975)) &
				   (df['weight'] >= df['weight'].quantile(0.025)) &
				   (df['weight'] <= df['weight'].quantile(0.975))]

thrown_amount = df_clean_data.shape[0]
total_amount = df.shape[0]

percent_thrown = (thrown_amount/total_amount)

print("The percentage of thrown data is %f" % percent_thrown)

# We got that that the percentage of thrown data is 9.037% of the original

# Part 2. Visual data analysis

# 2.1 Correlation matrix visualization

# We charge the data again

df2 = pd.read_csv("./Dataset/mlbootcamp5_train.csv")
pearsons_corr = df2.corr(method='pearson')
sns.heatmap(pearsons_corr)
plt.show()

# Using the correlation matrix we can appreciate that the features that keeps stronger correlation with gender feature are smoke and height

# 2.2 Height distribution of men and women

# We make a violin plot to study the distribution of height values for men and women

sns.violinplot(x='gender', y='height', data=df2)
plt.show()

# 2.3 Rank correlation

# Calculate the correlation matrix, but this time using Spearman's method

spearman_corr = df2.corr(method='spearman')
sns.heatmap(spearman_corr)
plt.show()

# 2.4 Age

# We study the age distribution in those who has CVD and those who don't through a countplot, where the x axis shows the age values and the y 
# axis shows the amount of people with that age. This plot should show 2 bars per x value, one for those with CVD and one for the rest.

sns.countplot(x='age_years', hue='cardio', data=df)
plt.show()

