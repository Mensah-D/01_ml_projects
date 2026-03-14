
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


# Checking for missing values in single column

col = "Cabin"
missing_pct = train[col].isna().mean()*100
print(f"{col}: {missing_pct: 2f}% missing")


# Check % of missing values 

missing_pct_by_col = (train.isna().mean() * 100).sort_values(ascending=False)
missing_pct_by_col

# Drop unusable columns from table

df = train.drop(columns=["Cabin", "PassengerId", "Name"])

df = df.drop(columns=["Ticket"])


# Create new column mapping sex + embarked to numberical value

df["Sex_mapped"] = df["Sex"].map({"male" : 0, "female" : 1})

df["Embarked_mapped"] = df["Embarked"].map({"S" : 0, "C" : 1, "Q" : 2})


# Dropping mapped columns from table

df = df.drop(columns=["Sex", "Embarked"])
df.head(10)

train.head(10)

# Plot number of people who survived over age and passenger class

survivors = df[df["Survived"] == 1].dropna(subset=["Age"]).copy()

bins = [0,10,20,30,40,50,60,70,80]
labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) -1)]
survivors["AgeGroup"] = pd.cut(survivors["Age"], bins=bins, labels=labels, right=False)
counts = survivors.groupby(["AgeGroup", "Pclass"]).size().unstack(fill_value=0)

counts.plot(kind="bar")
plt.title("Number of Survivors by Age Group and Passenger Class")
plt.xlabel("Age group")
plt.ylabel("Number of survivors")
plt.legend(title="Pclass")
plt.tight_layout()
plt.show()


# Plot correlation between features and label

label = "Survived"
numeric_df = df.select_dtypes(include=["number"])
corr_with_label = numeric_df.corr()[label].sort_values(key=lambda s: s.abs(), ascending=False)

corr_with_label = corr_with_label.drop(label)

plt.figure(figsize=(8, 4))
corr_with_label.plot(kind="bar")
plt.title(f"Correlation of numeric features with {label}")
plt.xlabel("Feature")
plt.ylabel("Correlation")
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


#Prepare Test Data

test_passenger_id = test["PassengerId"].copy()
test_df = test.drop(columns=["Cabin", "PassengerId", "Name", "Ticket"])
test_df["Sex_mapped"] = test_df["Sex"].map({"male" : 0, "female": 1})
test_df["Embarked_mapped"] = test_df["Embarked"].map({"S": 0, "C": 1, "Q":2})
test_df = test_df.drop(columns=["Sex", "Embarked"])

#Split features/target
X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_val, y_train, y_val = train_test_split(X, y , test_size=0.2, random_state=42, stratify=y)

