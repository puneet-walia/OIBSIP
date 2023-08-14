#<u>Oasis Infobyte Project</u>
##<i>Iris Flower Classification</i><br>
#"task 1"
###Puneet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = sns.load_dataset("iris")

dataset.head()

# features
x = dataset.iloc[:,1:-1].values
# answer values
y = dataset.iloc[:,-1].values

print("Features of dataset👇🏻\n")
for i in range(11):
  print(x[i])

print(" ")
print("Dependant Variables🌸\n")
for i in range(11):
  rv = np.random.choice(y)
  print(rv)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.svm import SVC
model = SVC()
model.fit(x_train,y_train)

predictions = model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,predictions)
print("Accuracy is : ", accuracy*100,"%")

count_setosa = 0
count_virginica = 0
count_versicolor = 0

for i in range(len(y)):
  if(y[i] == 'setosa'):
    count_setosa = count_setosa + 1
  elif(y[i] == 'virginica'):
    count_virginica = count_virginica + 1
  else:
    count_versicolor = count_versicolor + 1

categories = ['Setosa', 'Virginica', 'Versicolor']
values = [count_setosa,count_virginica,count_versicolor]

# Create the pie chart
plt.pie(values, labels=categories, autopct='%1.1f%%', startangle=90)

# Add a title to the pie chart
plt.title("Distribution of Categories")

# Display the legend
plt.legend()

# Show the pie chart
plt.show()

res = sns.barplot(y='petal_length',x='species',data=dataset,palette='rocket')
plt.show()

sns.stripplot(x='petal_length',y='petal_width',data=dataset,hue='species',palette='rocket')
custom_x_limits = (7, 30)
plt.xlim(custom_x_limits)

# Add a title to the plot
plt.title("Petal Length vs. Petal Width by Species")
plt.show()

print("Petal Length vs. Petal Width by Species")
sns.jointplot(x='petal_length',y='petal_width',data=dataset,hue='species',palette='rocket')
# custom_x_limits = (7, 30)
# plt.xlim(custom_x_limits)

# Add a title to the plot
plt.show()

sns.jointplot(x='petal_length',y='petal_width',data=dataset,kind='kde',palette='rocket',fill=True,cmap='rocket')
# Add a title to the plot
plt.show()

sns.pairplot(dataset,hue='species')

sns.heatmap(dataset.corr(),annot=True,cmap='rocket')
