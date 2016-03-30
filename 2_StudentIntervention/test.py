import numpy as np
import pandas as pd
student_data = pd.read_csv("student-data.csv")

feature_cols = list(student_data.columns[:-1])  # all columns but last are features
target_col = student_data.columns[-1]  # last column is the target/label
print "Feature column(s):-\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

X_all = student_data[feature_cols]  # feature values for all students
y_all = student_data[target_col]  # corresponding targets/labels
print "\nFeature values:-"
print X_all.head() 


def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX

X_all = preprocess_features(X_all)
print "Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns))


from sklearn.cross_validation import StratifiedShuffleSplit

# First, decide how many training vs test samples you want
num_all = student_data.shape[0]  # same as len(student_data)
num_train = 300  # about 75% of the data
num_test = num_all - num_train

cv = StratifiedShuffleSplit(student_data['passed'], 1, test_size  = num_test, random_state = 42)

for train_index, test_index in cv:
    X_train = X_all.iloc[train_index]
    y_train = y_all.iloc[train_index]
    X_test = X_all.iloc[test_index]
    y_test = y_all.iloc[test_index]
print "Training set: {} samples".format(X_train.shape[0])
print "Test set: {} samples".format(X_test.shape[0])
# Note: If you need a validation set, extract it from within training data


from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

#Ensemble methods
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

s =200
r = []
for i in range (1,100,1):
    #clf = RandomForestClassifier(random_state = i, n_estimators=10) #happens
    #clf = ExtraTreesClassifier(random_state = i, n_estimators=300) #hapens
    clf = DecisionTreeClassifier(random_state = i) #hapens



    #clf = KNeighborsClassifier()  doesnt happen
    #clf = GaussianNB()  doesnt happen
    #clf = svm.SVC(random_state = i) #has random state but always produces the same result, no matter the value
    #clf = AdaBoostClassifier() #has random state but always produces the same result, no matter the value
    #clf = GradientBoostingClassifier(random_state = 123)#has random state but always produces the same result, no matter the value

    
        
        
    clf.fit(X_train[:s], y_train[:s])        
    train_predictions = clf.predict(X_train[:s])        
    test_predictions = clf.predict(X_test)
    
    
    trainF1 =  f1_score(train_predictions, y_train[:s], pos_label='yes')
    testF1 =  f1_score(test_predictions, y_test, pos_label='yes')
    
    #print trainF1, testF1
    r.append(testF1)
    print testF1

print ''
print min(r), max(r), max(r) - min(r)

#RandomForestClassifier    
n_estimators,   min(F1Score),  max(F1Score),   diff
10:             0.645161290323 0.808823529412  0.16
50:             0.728571428571 0.805555555556  0.07
100:            0.719424460432 0.8             0.08
150:            0.744827586207 0.797202797203  0.05
200:            0.751773049645 0.797202797203  0.04
250:            0.75           0.797202797203  0.04
300:            0.753424657534 0.791666666667  0.03
1000:           0.753424657534 0.794520547945  0.04

#ExtraTreesClassifier
10:             0.614173228346 0.779411764706  0.16      
100:            0.715328467153 0.794326241135  0.07
300:            0.719424460432 0.769230769231  0.04


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        