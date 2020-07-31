# titanic_prediction
Kaggle competition: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck


# Data Cleaning and Feature Engineering

Before beginning the analysis, first try to understand the data. There are 12 columns in our dataframe:

**"Survived"** - dependent variable and what we are trying to predict, binary datatype of 1 for survived and 0 for did not survive

**"PassengerID"** and **"Ticket"** - unique identifiers, probably don't help predict outcome and will therefore be dropped 

**"Pclass"** - ticket class with 1 = upper class, 2 = middle class, and 3 = lower class

**"Name"** - represents passenger name, could potentially be parsed to get useful features, so I'll keep it

**"Sex"** - categorical variable, either male or female, could be converted into numerical variable for analysis

**"Age"** and **"Fare"** - both continuous variables

**"SibSp"** - number of related siblings/spouse aboard

**"Parch"** - number of related parents/children aboard

**"Cabin"** - approximate position on ship when incident occured

**"Embarked"** - categorical variable for port where passenger embarked, C = Cherbourg, Q = Queenstown, S = Southampton, could be converted into numerical variable for analysis


Create a new feature called FamilySize that is sum of SibSp and Parch

|   FamilySize|  Survived|
|---|---|
|0           |1|  0.303538|
|1           |2|  0.552795|
|2           |3 | 0.578431|
|3           |4  |0.724138|
|4           |5  |0.200000|
|5           |6  |0.136364|
|6          |7  |0.333333|
|7          |8  |0.000000|
|8          |11  |0.000000|


The size of the family seems to have an impact on the survival rates. From this I can create a further feature that I'll call 'IsAlone' that will be 1 if the passenger was alone on the ship and 0 otherwise.

Next, fill in any missing observations for 'Fare' with the median fare, then divide the category up into 5 different buckets for later analysis.

|   CategoricalFare|  Survived|
|---|---|
|(0, 7.91]  |0.197309|
|(7.91, 14.454]  |0.303571|
|(14.454, 31.0]  |0.454955|
|(31.0, 512.329]  |0.581081|

There seems to be a pretty clear connection between fare price and survival rate. Passengers who paid more have much high survival rates.

Since there are many missing values for Age, fill in the missing values by generating random numbers that are within a standard deviation of the mean. Then categorize age into 5 buckets

|  CategoricalAge|  Survived|
|---|---|
|  (-0.08, 16.0]|  0.512605|
|   (16.0, 32.0]|  0.359551|
|   (32.0, 48.0]|  0.364372|
|   (48.0, 64.0]|  0.434783|
|   (64.0, 80.0]|  0.090909|

Survival is a lot higher for children than for other groups, and the elderly have by far the lowest survival rates

While the names themselves probably won't tell us much, perhaps we can extract the titles from the names which might help

|   |female|  male|
|---|---|---|
|Capt           |0|     1|
|Col            |0|     2|
|Countess       |1|     0|
|Don            |0|     1|
|Dr             |1|     6|
|Jonkheer       |0|     1|
|Lady           |1|     0|
|Major          |0|     2|
|Master         |0|    40|
|Miss         |182|     0|
|Mlle           |2|     0|
|Mme            |1|     0|
|Mr             |0|   517|
|Mrs          |125|     0|
|Ms             |1|     0|
|Rev            |0|     6|
|Sir            |0|     1|

Now try to categorize these. Mlle is an abbreviation for Mademoiselle so should be grouped in with Miss. Mme is an abbreviation for Madame so should be grouped in with Mrs. Then create a broader category of high status titles, like Countess, Don, Dr, Jonkheer, etc. Then anything that is either not common (like Mr. or Mrs.) or doesn't fit into these groupings should go into a misc category (like Rev or Major)

Now, I'll remove any features that are not useful. "PassengerID" and "Ticket" likely aren't relevant and most of the values in the 'Cabin' are null so I'll drop all three. "SibSp" and "Parch" have been combined to get "FamilySize" so drop those two as well. Then since "Name" was used to get "Title", it too can be dropped

# Model Selection and Training
Because this is a classification problem, using a boosting algorithm seems appropriate. I'll use XGBoost

First start with a test/train split

<pre><code>
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), train['Survived'], random_state=100)

import xgboost as xgb

xgb = xgb.XGBClassifier(max_depth=10,learning_rate=0.005,n_estimators=500,min_child_weight=2)
xgb.fit(X_train,y_train)
</code></pre>

To get the accuracy of the model, use sklearn accuracy metrics. Finally export the predictions to a csv





