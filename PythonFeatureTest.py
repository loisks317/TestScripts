# standard import list 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# read in data
df=pd.read_csv('State_of_New_York_Mortgage_Agency.csv')

# for the sake of this script, make it just columns that could
# be regressed on
dfmod=df[['Original Loan Amount', 'Purchase Year', 'Original Loan To Value', 'SONYMA DPAL/CCAL Amount', 'Number of Units', 'Household Size ']]
dfmod=dfmod.replace('[\$,]', '', regex=True)
dfmod=dfmod.replace('[\%,]', '', regex=True)
dfmod=dfmod.replace('Family', '', regex=True)
dfmod=dfmod.astype(float)

# test for correlations 
corrDF=dfmod.corr()

# write out a scatter plot that visualizes this
# normalize everything by the max value so we can put
# everything on one plot 
dfnorm= dfmod.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# plot the norms
for ii in dfnorm.columns.values:
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_title(ii)
    # maybe think if we want to assign one color or a different window. I think 
    # I want to assign a different window 
    filtered = [ v for v in dfnorm.columns.values if not v.startswith(ii) ]
    colormap = plt.cm.gist_ncar
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(filtered))])

    for jj in filtered: 
       ax.scatter(dfnorm[ii], dfnorm[jj], s=30, alpha=0.3 )
       
    plt.legend(filtered, bbox_to_anchor=(1.1, 1.1))
    plt.savefig(ii[:5] + '_to_'+jj[:5]+'.png')

#
# so we've determined that there aren't any strong correlations between 
# our features. The 0.6 between Original Loan Amount and SONYMA DPAL/CCAL Amount
# is concerning and something to be aware of, but not show stopping. Let's
# add another variable that does highly correlate so we can see the effects
# We will call it loan year... when you first requested the loan? 
#
randoms = np.linspace(0.9, 1.1, len(dfmod))
dfmod['Loan Year']=dfmod['Purchase Year']*randoms
corrDF=dfmod.corr(); print(corrDF)

#
# OK, being naive data scientists, now we're going to try to model the Original Loan Amount based on all the other features 
#
dfmod=dfmod.dropna()
result=dfmod['Original Loan Amount']
data=dfmod.drop(['Original Loan Amount'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data , result , test_size=0.25, random_state=1)

# setup the model
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

clf = RandomForestRegressor()
clf.fit(X_train, y_train)
predictions=clf.predict(X_test)
print('WITH MADE UP DATA')
print(metrics.r2_score(predictions, y_test))
print(np.sort(list(zip(data.columns.values, clf.feature_importances_)))[::-1])

#
# repeat without the made up column
print('WITHOUT MADE UP DATA')
data2=dfmod.drop(['Original Loan Amount', 'Loan Year'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data2 , result , test_size=0.25, random_state=1)

# setup the model
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

clf2 = RandomForestRegressor()
clf2.fit(X_train, y_train)
predictions=clf2.predict(X_test)
print(metrics.r2_score(predictions, y_test))
print(np.sort(list(zip(data2.columns.values, clf2.feature_importances_)))[::-1])

#
# Interesting, right? the r^2 of the made up data model was better than the model 
# without made up data. Uh oh! 
# Why did this happen? Well, let's try to see if we can engineer a few more features
# to develop a better model
#
# let's add some financial data
# our purchase years range from 2004 to 2016. Let's get the average mortage rate in those years
# googled and found at : http://www.freddiemac.com/pmms/pmms30.html
years=np.linspace(2004., 2016., 13)
mort30=np.array([5.84, 5.87, 6.41, 6.34, 6.03, 5.04, 4.69, 4.45, 3.66, 3.98, 4.17, 3.85, 3.65])
data['mort']=[mort30[years==data['Purchase Year'].iloc[item]][0] for item in range(len(data['Purchase Year']))]
data2['mort']=[mort30[years==data2['Purchase Year'].iloc[item]][0] for item in range(len(data2['Purchase Year']))]

#
# what else can we add? Maybe how many houses were purchased in that year 
# from https://www.statista.com/statistics/219963/number-of-us-house-sales/
housesBought=np.array([1203, 1283, 1051, 776, 485, 375, 323, 306, 368, 429, 437, 501, 560])*1000.
data['housesBought']=[housesBought[years==data['Purchase Year'].iloc[item]][0] for item in range(len(data['Purchase Year']))]
data2['housesBought']=[housesBought[years==data2['Purchase Year'].iloc[item]][0] for item in range(len(data2['Purchase Year']))]

#
#  and just because we don't want all our new data depending on year, let's do one about
#  expected wealth by family size in NY
#  source: https://www.justice.gov/ust/eo/bapcpa/20130501/bci_data/median_income_table.htm
#  assume > 4 is = 4
familySize=np.sort(data['Household Size '].unique())
income=[47790, 59308,69052, 83209]
for i in range(len(income),len(familySize)):
    income.append(83209)
income=np.array(income)
data['income']=[income[familySize==data['Household Size '].iloc[item]][0] for item in range(len(data['Household Size ']))]
data2['income']=[income[familySize==data2['Household Size '].iloc[item]][0] for item in range(len(data2['Household Size ']))]

stop

# now go back to model !

X_train, X_test, y_train, y_test = train_test_split(
    data , result , test_size=0.25, random_state=1)

# setup the model

clf3 = RandomForestRegressor()
clf3.fit(X_train, y_train)
predictions=clf3.predict(X_test)
print('WITH MADE UP DATA')
print(metrics.r2_score(predictions, y_test))
print(np.sort(list(zip(data.columns.values, clf3.feature_importances_)))[::-1])

#
# repeat without the made up column
print('WITHOUT MADE UP DATA')
data2=dfmod.drop(['Original Loan Amount', 'Loan Year'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    data2 , result , test_size=0.25, random_state=1)

# setup the model
clf4 = RandomForestRegressor()
clf4.fit(X_train, y_train)
predictions=clf4.predict(X_test)


print(metrics.r2_score(predictions, y_test))
print(np.sort(list(zip(data2.columns.values, clf4.feature_importances_)))[::-1])