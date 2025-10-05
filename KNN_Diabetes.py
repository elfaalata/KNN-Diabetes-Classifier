import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


#Loading the data as df
df = pd.read_csv("diabetes_dataset.csv")

print(df.head(5)) #Gives a look into the df in order to gain some insight on it
print(df.info(), "\n") #Helps identify any missing data and summarizes df info

#We can see there arent any missing data or entries so we dont need to handle anything
#We will build a simple KNN classifier model to predict diabetes using age as our feature type and diagnosed diabetes as our target variable

#We will check the diagnosed diabetes column and see how many patients are diagnosed with diabeters/not 
print(df["diagnosed_diabetes"].value_counts(normalize=True), "\n") #As a percentage

#The output shows us that 0.6% of patients are diagnosed meanwhile 0.4% are not, this tells us that the data is slightly skewed to diagnosing yes
#We'll now look at the age column

print(df["age"].describe(), "\n")

below_0 = df["age"] < 0
above_100 = df['age'] > 100
print(below_0.sum())
print(above_100.sum(), "\n")

#Since there are no invalid or extreme outliers, we can visualize a histogram of the age distribution
#Lets visualize this onto a histogram

plt.hist(df["age"], bins=10, edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

#Now lets define the feature and target
x = df[["age"]]
y = df["diagnosed_diabetes"]

#Since KNN uses distance, its important to scale and standardize our ages
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#Now we split the data into a training data set and a testing data set
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.2, random_state = 31, stratify = y)
#This splits our data into a test and train data, with the test size as 20% since our data is medium sized not too small or large 
#We included stratify since our data is slightly imbalanced and we want the split to preserve the original distribution

#Now we train the KNN
knn = KNeighborsClassifier(n_neighbors = 5) #We'll start with 5 neighbors
knn.fit(x_train, y_train)

#Now we let the KNN make predictions
y_predict = knn.predict(x_test)
print(y_predict) #Lets see the predictions

#Lets check the accuracy
accuracy = accuracy_score(y_test, y_predict)
print(accuracy, "\n")
#This returns a percentage of roughly 56%, which is not high, this can be due to many things such as age not being a strong factor, needing more features, or by fine-tuning the k value we could get better results
#Lets use more features and see if we can improve the accuracy

features = ["age", "bmi", "physical_activity_minutes_per_week", "family_history_diabetes", "glucose_fasting", "glucose_postprandial"]
x = df[features]

#Scale the features once again
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=31, stratify=y)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

#Make the predictions again
y_predict = knn.predict(x_test)
print(y_predict)

#Check accuracy again
accuracy = accuracy_score(y_test, y_predict)
print(accuracy, "\n")

#We can see that the accuracy improved but only ever so slightly, lets see if we can fine-tune the k value
accuracies = {}
for k in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[k] = acc  #Store k as key and accuracy as value
    print(f"k={k}, Accuracy={accuracy_score(y_test, y_pred)}")
print("")
#We know now that the most accurate k value is 10, with a percentage of roughly 83.6%

#Lets display a report of the model
print(classification_report(y_test, y_pred), "\n")
#This shows we achieved a KNN classifier with 84% accuracy, this model is balanced and generalizes well for it being a simple one

#Now we can plot an accuracy curve to show k vs accuracy
plt.plot(list(accuracies.keys()), list(accuracies.values()), marker='o')
plt.title('KNN Model Accuracy by K Value')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()