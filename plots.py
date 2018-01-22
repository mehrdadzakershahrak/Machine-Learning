import pandas as pd
import matplotlib.pyplot as plt

datapath = 'output3.csv'
plotpath = 'plots/'

'''
load data
'''
print '\nloading data...'

Xy = pd.read_csv(datapath, header = None, skiprows = 1, usecols = range(0,69)) # y is column at index = 2, X is the remaining columns

# seperate feature (X) and output (y) matrices from full data matrix
y = Xy.iloc[:, 2:3] # extract column at index 2, contains citations
del Xy[2]
X = Xy # name change for clarity

# extract relevant features
paperindices = X.iloc[:, 0:1]
authorcitation = X.iloc[:, 1:2]
year = X.iloc[:, 2:3]
pubcitation = X.iloc[:, 3:4]

'''
histograms of features and y with statistical info
'''
print '\nhistogram construction...'

# citations (y)
plt.close('all')

plt.title('Distribution of Citations')
plt.ylabel('Frequency')
plt.xlabel('Citations per Paper')
plt.axis([0, 3000, 0, 22000])
plt.hist(y.values, bins = 1500, color = 'blue', edgecolor='none')
plt.figtext(1.0, 0.2, y.describe())

plt.savefig(plotpath + 'citationshistogram.png', bbox_inches = 'tight') # bbox_inches = 'tight' makes it so text to side of plot is also recorded

# author_citation
plt.close('all')

plt.title('Distribution of Author_Citation')
plt.ylabel('Frequency')
plt.xlabel('Author Citations per Paper')
plt.axis([0, 100000, 0, 80000])
plt.hist(authorcitation.values, bins = 1500, color = 'blue', edgecolor='none')
plt.figtext(1.0, 0.2, authorcitation.describe())

plt.savefig(plotpath + 'authorcitationhistogram.png', bbox_inches = 'tight')

# pub_citation
plt.close('all')

plt.title('Distribution of Pub_Citation')
plt.ylabel('Frequency')
plt.xlabel('Publication Venue Citations per Paper')
plt.axis([0, 325000, 0, 60000])
plt.hist(pubcitation.values, bins = 1500, color = 'blue', edgecolor='none')
plt.figtext(1.0, 0.2, pubcitation.describe())

plt.savefig(plotpath + 'pubcitationhistogram.png', bbox_inches = 'tight')

# year
plt.close('all')

plt.title('Distribution of Year')
plt.ylabel('Frequency')
plt.xlabel('Year of Paper Publication')
plt.axis([1930, 2015, 0, 400000])
plt.hist(year.values, bins = 1500, color = 'blue', edgecolor='none')
plt.figtext(1.0, 0.2, year.describe())

plt.savefig(plotpath + 'yearhistogram.png', bbox_inches = 'tight')

'''
scatter plots of y vs features
'''
print '\nscatter plot construction...'

# citations vs index (not a feature)
plt.close('all')

fig, ax = plt.subplots()
ax.set_title('Citations vs Paper Index')
ax.set_ylabel('Number of Citations per Paper')
ax.set_xlabel('Index of Paper')
ax.scatter(paperindices, y, c = 'r')

plt.savefig(plotpath + 'citationsvsindex.png', bbox_inches = 'tight') # bbox_inches = 'tight' makes it so text to side of plot is also recorded

# citations vs author_citation
plt.close('all')

fig, ax = plt.subplots()
ax.set_title('Citations vs Author_Citation')
ax.set_ylabel('Number of Citations per Paper')
ax.set_xlabel('Author Citations per Paper')
ax.scatter(authorcitation, y, c = 'r')

plt.savefig(plotpath + 'citationsvsauthorcitations.png', bbox_inches = 'tight')

# citations vs pub_citation
plt.close('all')

fig, ax = plt.subplots()
ax.set_title('Citations vs Pub_Citation')
ax.set_ylabel('Number of Citations per Paper')
ax.set_xlabel('Publication Venue Citations per Paper')
ax.scatter(pubcitation, y, c = 'r')

plt.savefig(plotpath + 'citationsvspubcitations.png', bbox_inches = 'tight')

# citations vs year
plt.close('all')

fig, ax = plt.subplots()
ax.set_title('Citations vs Year')
ax.set_ylabel('Number of Citations per Paper')
ax.set_xlabel('Year of Paper Publication')
ax.set_xlim([1930, 2015])
ax.scatter(year, y, c = 'r')

plt.savefig(plotpath + 'citationsvsyear.png', bbox_inches = 'tight')

'''
classification results
'''
print '\nclassification plots...'

classifiers = ['kNN', 'Logistic Regression', 'Decision Tree']
accuracies = [73.6, 70, 70.5]

plt.close('all')

fig, ax = plt.subplots()
plt.ylabel('Average Classification Accuracy (%)')
ax.bar(classifiers, accuracies, color = ['lavender', 'lightblue', 'forestgreen'])

plt.savefig(plotpath + 'classificationaccuracy.png', bbox_inches = 'tight')