#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 13:23:26 2026

@author: dennismensah
"""

# Importing Library
import pandas as pd

project_url = 'https://raw.githubusercontent.com/gimseng/99-ML-Learning-Projects/'
data_path = 'master/001/data/'
train = pd.read_csv(project_url+data_path+'train.csv')
test = pd.read_csv(project_url+data_path+'test.csv')
train.head()
test.head()

train.isna().sum()
test.isna().sum()

train.describe()

# Importing matplotlib library
import matplotlib.pyplot as plt

#defining gender distribution data

gender_distribution = train["Sex"].value_counts()

gender_distribution.plot(kind="bar")
plt.title("Gender distribution on board")
plt.xlabel("Gender")
plt.ylabel("Number of passengers")
plt.show()

# Survival By Gender

survival_by_gender = train.groupby("Sex")["Survived"].mean()

survival_by_gender.plot(kind="bar")
plt.title("Survival rate by gender")
plt.xlabel("Gender")
plt.ylabel("Survival rate")
plt.ylim(0,1)
plt.show()

# Survived vs not survived

gender_survival_counts = pd.crosstab(train["Sex"], train["Survived"])
gender_survival_counts.columns = ["Did not survive", "Survived"]

gender_survival_counts.plot(kind="bar")
plt.title("Survived vs not survived by gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# Survival rate by Pclass (Males only)

male_survival_by_pclass = train[train["Sex"] == "male"].groupby("Pclass")["Survived"].mean().sort_index()
male_survival_by_pclass.plot(kind="bar")
plt.title("Survival rate by Pclass (Males only)")
plt.xlabel("Pclass")
plt.ylabel("Survival rate")
plt.ylim(0,1)
plt.show()


# Survival rate by Pclass (females only)

female_survival_by_pclass = train[train["Sex"] == "female"].groupby("Pclass")["Survived"].mean().sort_index()

female_survival_by_pclass.plot(kind="bar")
plt.title("Survival rate by Pclass (Females only)")
plt.xlabel("Pclass")
plt.ylabel("Survival rate")
plt.ylim(0,1)
plt.show()


col = "Cabin"
missing_pct = train[col].isna().mean()*100
print(f"{col}: {missing_pct: 2f}% missing")

