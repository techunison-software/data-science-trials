
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer

dataset = pd.read_csv('E:/LINEARREGRESSION/Vijay/Titanic Dataset/INPUT/train.csv')
test_data = pd.read_csv('E:/LINEARREGRESSION/Vijay/Titanic Dataset/INPUT/test.csv')

# Histogram

dataset['Age'].hist()
plt.show()

fig=plt.figure() #Plots in matplotlib reside within a figure object, use plt.figure to create new figure
#Create one or more subplots using add_subplot, because you can't create blank figure
ax = fig.add_subplot(1,1,1)
#Variable
ax.hist(dataset['Pclass'],bins = 30) # Here you can play with number of bins
# Labels and Tit
plt.title('Fare distribution')
plt.xlabel('Pclass')
plt.ylabel('Fare')
plt.show()

#Column Chart

dataset.plot.bar()   
plt.bar(dataset['PassengerId'], dataset['Pclass']) 
plt.xlabel("PassengerId") 
plt.ylabel("Pclass") 
plt.show() 

# Box Plot Chart

# For each numeric attribute of dataframe 
dataset.plot.box()   
plt.boxplot(dataset['Survived'])  # individual attribute box plot 
plt.show()

# Box Plot 

fig=plt.figure()
ax = fig.add_subplot(1,1,1)
#Variable
ax.boxplot(dataset['Age'])
plt.show()

# #Violin Chart

sns.violinplot(dataset['Age'], dataset['Sex']) #Variable Plot
sns.despine()
plt.show()

#Pie Chart

var=dataset.groupby(['Sex']).sum().stack()
temp=var.unstack()
type(temp)
x_list = temp['Age']
label_list = temp.index
# pyplot.axis("equal") #The pie chart is oval by default. To make it a circle use pyplot.axis("equal")
plt.pie(x_list,labels=label_list,autopct="%1.1f%%") #To show the percentage of each pie slice, pass an output format to the autopctparameter 
plt.title("Male and Female Ratio") 
plt.show()


# bubble chart

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(dataset['Age'],dataset['Sex'], s=dataset['Pclass']) # Added third variable income as size of the bubble
plt.show()


# Scatter Plot

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(dataset['Age'],dataset['Sex']) #You can also add more variables here to represent color and size.
plt.show()


# Stacked Column Chart

var = dataset.groupby(['Age','Sex']).PassengerId.sum()
var.unstack().plot(kind='bar',stacked=True,  color=['red','blue'], grid=False)
plt.show()


# Line Chart

var = dataset.groupby('Age').PassengerId.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('PassengerId')
ax1.set_ylabel('Fare')
ax1.set_title("Passenger wise fare of Travel")
var.plot(kind='line')
plt.show()


# Bar Chart

var = dataset.groupby('Sex').Fare.sum() #grouped sum of sales at Gender level
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('PassengerId')
ax1.set_ylabel('Fare')
ax1.set_title("Passenger wise fare of Travel")
var.plot(kind='bar')
plt.show()

var = dataset.groupby('Sex').Fare.sum() #grouped sum of sales at Gender level
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Sex')
ax1.set_ylabel('Fare')
ax1.set_title("Gender wise Fare of Travel")
var.plot(kind='bar')
plt.show()


# Heat Map

#Generate a random number, you can refer your data values also
data = np.random.rand(4,2)
rows = list('1234') #rows categories
columns = list('MF') #column categories
fig,ax=plt.subplots()
#Advance color controls cmap=plt.cm.Blues
ax.pcolor(data,cmap=plt.cm.Reds,edgecolors='k')
ax.set_xticks(np.arange(0,2)+0.5)
ax.set_yticks(np.arange(0,4)+0.5)
# Here we position the tick labels for x and y axis
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
#Values against each labels
ax.set_xticklabels(columns,minor=False,fontsize=20)
ax.set_yticklabels(rows,minor=False,fontsize=20)
plt.show()



