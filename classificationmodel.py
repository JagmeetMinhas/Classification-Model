#Import dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

#Read in the csv data
df = pd.read_csv("Restaurant_Dataset.csv")

#Drop any NaNs
df = df.dropna()

#Display a graph of the top 5 currencies represented
currencyCounter = df['Currency'].value_counts().nlargest(5)
plt.figure(figsize=(15, 10))
sns.barplot(x=currencyCounter.index, y=currencyCounter.values, palette="viridis")
plt.title('Count of Top 5 Currencies')
plt.xlabel('Currency Type')
plt.ylabel('Count')
plt.show()

#Display a pie chart with the rating distribution
ratingCounter = df["Rating text"].value_counts()
ratingCounter.plot.pie(figsize=(8, 8))
plt.ylabel('')
plt.title('Rating Distribution')
plt.show()

#Displaying a histogram of the numerical ratings
df['Aggregate rating'].plot.hist(bins=5)
plt.xlabel('Rating')
plt.title('Numerical Rating Distribution')
plt.show()

#Dropping unecessary columns
df = df.drop("Restaurant ID", axis = 1)
df = df.drop("Restaurant Name", axis = 1)
df = df.drop("Country Code", axis = 1)
df = df.drop("City", axis = 1)
df = df.drop("Address", axis = 1)
df = df.drop("Locality", axis = 1)
df = df.drop("Locality Verbose", axis = 1)
df = df.drop("Cuisines", axis = 1)
df = df.drop("Currency", axis = 1)
#These are being dropped since I wish to use the textual rating as my classifier
df = df.drop("Aggregate rating", axis = 1)
df = df.drop("Rating color", axis = 1)

#Encoding the boolean columns (Has Table booking, Has Online delivery, etc) with 1s and 0s
df.replace({'Yes': 1, 'No': 0}, inplace = True)

#Removing the "Not Rated" rows from the text rating
df = df[df['Rating text'] != "Not rated"]

#Creating the x and y value sets
xValues = df[["Average Cost for two", "Has Online delivery", "Has Table booking", "Is delivering now", "Switch to order menu", "Price range"]].values
yValues = df[["Rating text"]].values

#Creating the training and testing sets
xTraining, xTesting, yTraining, yTesting = train_test_split(xValues, yValues, test_size = 0.30, random_state = 0)

#Rescaling the columns
scalingObj = StandardScaler()
xTraining = scalingObj.fit_transform(xTraining)
xTesting = scalingObj.transform(xTesting)

#Creating the KNN model
knnModel = KNeighborsClassifier(n_neighbors=5)
knnModel.fit(xTraining, yTraining.ravel())

#Creating the Naive Bayes Model
bayesianModel = GaussianNB()
bayesianModel.fit(xTraining, yTraining.ravel())

#Creating a radial basis function SVM model (this one performed the best, and training the other SVM models severely impacted runtime)
radialSVM = svm.SVC(kernel='rbf', gamma=2, C=1, decision_function_shape='ovo', probability=True).fit(xTraining, yTraining.ravel())

#Creating a decision tree classifier
dTree = DecisionTreeClassifier()
dTree.fit(xTraining, yTraining)

#Creating a random forest classifier
rForest = RandomForestClassifier(n_estimators=10, criterion='gini',random_state=1)
rForest.fit(xTraining, yTraining.ravel())

#Creating lists for the cross validation accuracy graph
accList = []
modelList = []

#Performing 5 fold cross validation for each model
crossValidation = KFold(n_splits=5)

result = cross_val_score(knnModel, xTraining, yTraining.ravel(), cv = crossValidation, scoring='accuracy')
print(f'Avg accuracy for KNN: {result.mean()}')
accList.append(result.mean())
modelList.append("KNN")

result = cross_val_score(bayesianModel, xTraining, yTraining.ravel(), cv = crossValidation, scoring='accuracy')
print(f'Avg accuracy for Naive Bayes: {result.mean()}')
accList.append(result.mean())
modelList.append("Naive Bayes")

result = cross_val_score(radialSVM, xTraining, yTraining.ravel(), cv = crossValidation, scoring='accuracy')
print(f'Avg accuracy for radial basis function support vector machines: {result.mean()}')
accList.append(result.mean())
modelList.append("SVM")

result = cross_val_score(dTree, xTraining, yTraining.ravel(), cv = crossValidation, scoring='accuracy')
print(f'Avg accuracy for decision trees: {result.mean()}')
accList.append(result.mean())
modelList.append("Decision Tree")

result = cross_val_score(rForest, xTraining, yTraining.ravel(), cv = crossValidation, scoring='accuracy')
print(f'Avg accuracy for random forest: {result.mean()}')
accList.append(result.mean())
modelList.append("Random Forest")

#Creating an accuracy graph
plt.figure(figsize = (15, 10))
plt.bar(modelList, accList)
plt.title('Model vs Accuracy',fontsize=17)
plt.xlabel('Model',fontsize=17)
plt.ylabel('Accuracy',fontsize=17)
plt.show()

#Making predictions
knnPred = knnModel.predict(xTesting)
bayesPred = bayesianModel.predict(xTesting)
svmPred = radialSVM.predict(xTesting)
dTreePred = dTree.predict(xTesting)
rForestPred = rForest.predict(xTesting)

#Generating classification reports
#NOTE: For some reason, without the zero_division parameter, I recieved a warning
#This is only for fulfilling the display requirement, I'm going to generate the report again but as a dictionary for the large graph
print(classification_report(yTesting, knnPred, zero_division=0))
print(classification_report(yTesting, bayesPred, zero_division=0))
print(classification_report(yTesting, svmPred, zero_division=0))
print(classification_report(yTesting, dTreePred, zero_division=0))
print(classification_report(yTesting, rForestPred, zero_division=0))

#Creating lists for the classication graph (the accuracy list has already been created, see above)
precList = []
recList = []
f1List = []

#Appending the neccessary values by outputting the report as a dictionary
knnDict = classification_report(yTesting, knnPred, zero_division=0, output_dict = True)
precList.append(knnDict['weighted avg']['precision'])
recList.append(knnDict['weighted avg']['recall'])
f1List.append(knnDict['weighted avg']['f1-score'])

bayesDict = classification_report(yTesting, bayesPred, zero_division=0, output_dict = True)
precList.append(bayesDict['weighted avg']['precision'])
recList.append(bayesDict['weighted avg']['recall'])
f1List.append(bayesDict['weighted avg']['f1-score'])

svmDict = classification_report(yTesting, svmPred, zero_division=0, output_dict = True)
precList.append(svmDict['weighted avg']['precision'])
recList.append(svmDict['weighted avg']['recall'])
f1List.append(svmDict['weighted avg']['f1-score'])

treeDict = classification_report(yTesting, dTreePred, zero_division=0, output_dict = True)
precList.append(treeDict['weighted avg']['precision'])
recList.append(treeDict['weighted avg']['recall'])
f1List.append(treeDict['weighted avg']['f1-score'])

forestDict = classification_report(yTesting, rForestPred, zero_division=0, output_dict = True)
precList.append(forestDict['weighted avg']['precision'])
recList.append(forestDict['weighted avg']['recall'])
f1List.append(forestDict['weighted avg']['f1-score'])

#Now that the lists have been created, I pasted in the code from the shared notebook, adjusting as necessary
n_bars = 5
n_groups = 4
total_width = 0.8
bar_width = total_width / n_groups
X_axis = np.arange(n_bars)
left_positions = X_axis - (total_width - bar_width) / 2
plt.rcParams["figure.figsize"] = (20, 3)
plt.bar(left_positions, accList, bar_width, label='Accuracy')
plt.bar(left_positions + bar_width, precList, bar_width, label='Precision')
plt.bar(left_positions + bar_width * 2, recList, bar_width, label='Recall')
plt.bar(left_positions + bar_width * 3, f1List, bar_width, label='F1')
plt.xticks(X_axis, ["KNN", "Bayes", "SVM", "DT", "RF"])
plt.xlabel("Model")
plt.ylabel("Weighted Avg")
plt.title("Model vs Weighted Avg Of Various Metrics")
plt.legend(loc='upper right')
plt.show()

#Exporting the Random Forest Model, as it performed the best
with open('Classification_Model_DT.pkl', 'wb') as f:
  pickle.dump(rForest, f)
