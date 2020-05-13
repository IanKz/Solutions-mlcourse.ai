import pandas as pd
import numpy as np

df = pd.read_csv("./Dataset/adult.data.csv")

# 1.- How many men and women (sex feature) are represented in this dataset

men = df[df['sex'] == 'Male'].shape[0]
women = df[df['sex'] == 'Female'].shape[0]

print("There are %d men and %d women." % (men, women))

# 2.- What is the average age (age feature) of women?

women_avg = df[df['sex'] == 'Female']['age'].describe()['mean']

print('The average age for women is %f' %  women_avg)

# 3.- What is the percentage of German citizens (native-country feature)

nationality_count = df['native-country'].value_counts(normalize=True)
german_percentage = nationality_count['Germany']

print("The percentage of German citizens is: %f" % german_percentage)

# 4-5.- What are the mean and standard deviation of age for those who earn more than 50k per year (salary feature) and those who earn less than 50K per year?

big_earners_mean = df[df['salary'].apply(lambda salary: salary == '>50K')]['age'].mean()
big_earners_std = df[df['salary'].apply(lambda salary: salary == '>50K')]['age'].std()
little_earners_mean = df[df['salary'].apply(lambda salary: salary == '<=50K')]['age'].mean()
little_earners_std = df[df['salary'].apply(lambda salary: salary == '<=50K')]['age'].std()

print("For people who earn more than 50K per year the mean is %f and the std deviation is %f" % (big_earners_mean, big_earners_std))
print("For people who earn less than 50K per year the mean is %f and the std deviation is %f" % (little_earners_mean, little_earners_std)) 

# 6.- Is it true that people who earn more than 50K have at least high school education? 
# (education – Bachelors, Prof-school, Assoc-acdm, Assoc-voc, Masters or Doctorate feature)

big_earners = df[df['salary'].apply(lambda salary: salary == '>50K')]

bachelors = big_earners[big_earners["education"].apply(lambda education: education == 'Bachelors')]['age'].count()
prof_school = big_earners[big_earners["education"].apply(lambda education: education == 'Prof-school')]['age'].count()
assoc_acdm = big_earners[big_earners["education"].apply(lambda education: education == 'Assoc-acdm')]['age'].count()
assoc_voc = big_earners[big_earners["education"].apply(lambda education: education == 'Assoc-voc')]['age'].count()
masters = big_earners[big_earners["education"].apply(lambda education: education == 'Masters')]['age'].count()
doctorate = big_earners[big_earners["education"].apply(lambda education: education == 'Doctorate')]['age'].count()

big_earners_education = bachelors + prof_school + assoc_acdm + assoc_voc + masters + doctorate == big_earners['age'].count()

print("The fact that people who earn more than 50K have at least high school education is", str(big_earners_education))

# 7.- Display age statistics for each race (race feature) and each gender (sex feature). Use groupby() and describe(). 
# Find the maximum age of men of Amer-Indian-Eskimo race.

age_statistics = df.groupby(["age"]).describe()

print("The age statistics are:")
print(age_statistics)

race_statistics = df.groupby(['race']).describe()

print("The race statistics are:")
print(race_statistics)

sex_statistics = df.groupby(['sex']).describe()

print("The sex statistics are:")
print(sex_statistics)

# 8.-Among whom is the proportion of those who earn a lot (>50K) greater: married or single men (marital-status feature)? 
# Consider as married those who have a marital-status starting with Married (Married-civ-spouse, Married-spouse-absent or Married-AF-spouse), 
# the rest are considered bachelors.

married = big_earners[big_earners['marital-status'].apply(lambda maritalS: maritalS[:7] == 'Married')]
not_married = big_earners[big_earners['marital-status'].apply(lambda maritalS: maritalS[:7] != 'Married')]

married_men = married[married['sex'].apply(lambda gender: gender == 'Male')]['age'].count()
not_married_men = not_married[not_married['sex'].apply(lambda gender: gender == 'Male')]['age'].count()

proportion = married_men/not_married_men

print("The proportion between those men that earn more than 50K and are married and thos that aren't married is " + str(proportion) + " to 1")

# 9.- What is the maximum number of hours a person works per week (hours-per-week feature)? 
# How many people work such a number of hours, and what is the percentage of those who earn a lot (>50K) among them?

max_hours = df['hours-per-week'].max()
people_max_hours = df[df['hours-per-week'].apply(lambda hours: hours == max_hours)];
amount_max_hours = people_max_hours['age'].count()
big_earners_max_hours = people_max_hours[people_max_hours['salary'].apply(lambda salary: salary == '>50K')]
amount_big_earners_max_hours = big_earners_max_hours['age'].count()
percentage_big_earners = amount_max_hours/amount_big_earners_max_hours

print("The maximum amount of hours that a person works per week is %f." % max_hours)
print("The amount of people that works that amount of hours per week is %d." % amount_max_hours)
print("The amount of big earners that works %f hours per week is %d, that represents the %f percent of all of them." % (max_hours, amount_big_earners_max_hours, percentage_big_earners))

# 10.- Count the average time of work (hours-per-week) for those who earn a little and a lot (salary) for each country (native-country). 
# What will these be for Japan?

big_earners = df[df['salary'].apply(lambda salary: salary == '>50K')]
little_earners = df[df['salary'].apply(lambda salary: salary == '<=50K')]

big_earners_avg_hours = big_earners['hours-per-week'].mean()
little_earners_avg_hours = little_earners['hours-per-week'].mean()

japanese_big_earners = big_earners[big_earners['native-country'].apply(lambda country: country == 'Japan')]
japanese_little_earners = little_earners[little_earners['native-country'].apply(lambda country: country == 'Japan')]

japanese_big_earners_avg_hours = japanese_big_earners['hours-per-week'].mean()
japanese_little_earners_avg_hours = japanese_little_earners['hours-per-week'].mean()

print("The average hours worked per hour by big earners is %f and for little earners is %f" % (big_earners_avg_hours, little_earners_avg_hours))
print("Particulary in japan for big earners is %f and for little earners is %f" % (japanese_big_earners_avg_hours, japanese_little_earners_avg_hours))

