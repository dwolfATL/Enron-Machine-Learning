#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Helper function to visualize data
def visualize_plot(data):    
    for point in data:
        index1 = point[1]
        index2 = point[4]
        plt.scatter( index1, index2 )
    plt.xlabel("Salary")
    plt.ylabel("Bonus")
    plt.show()
    return

# Create a list of all features (excluding email address)
# Also include the names of features that will be created
all_features = ['poi','salary','deferral_payments','total_payments',
                'loan_advances','bonus','restricted_stock_deferred',
                'deferred_income','total_stock_value', 'expenses',
                'exercised_stock_options','other','long_term_incentive',
                'restricted_stock','director_fees','to_messages',
                'from_poi_to_this_person','from_messages',
                'from_this_person_to_poi','shared_receipt_with_poi',
                'sent_to_poi_percent','rec_from_poi_percent',
                'payments_to_stocks']

# Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

# Remove two outliers that do not correspond to individual people
del data_dict['TOTAL']
del data_dict['THE TRAVEL AGENCY IN THE PARK']

### FEATURE CREATION
# Create a feature for percentage of emails received that were from a POI
for name in data_dict:
    if data_dict[name]['from_poi_to_this_person'] != "NaN" and \
        data_dict[name]['to_messages'] != "NaN":
        data_dict[name]['rec_from_poi_percent'] = \
        float(data_dict[name]['from_poi_to_this_person'])/ \
        float(data_dict[name]['to_messages'])
    else:
        data_dict[name]['rec_from_poi_percent'] = "NaN"
# Create a feature for percentage of emails sent that were to a POI
for name in data_dict:
    if data_dict[name]['from_this_person_to_poi'] != "NaN" and \
        data_dict[name]['from_messages'] != "NaN":
        data_dict[name]['sent_to_poi_percent'] = \
        float(data_dict[name]['from_this_person_to_poi'])/ \
        float(data_dict[name]['from_messages'])
    else:
        data_dict[name]['sent_to_poi_percent'] = "NaN"
# Create a feature for the ratio of payments to stocks
for name in data_dict:
    if data_dict[name]['total_payments'] != "NaN" and \
        data_dict[name]['total_stock_value'] != "NaN":
        data_dict[name]['payments_to_stocks'] = \
        float(data_dict[name]['total_payments'])/ \
        float(data_dict[name]['total_stock_value'])
    else:
        data_dict[name]['payments_to_stocks'] = "NaN"

# Store to my_dataset for easy export below.
my_dataset = data_dict

# Get a better understanding of the data by researching NaN values
# If less than 25% of the 144 entries in this dataset are valid
# entries, highlight them for removal
nan_dict = {}
remove_dict = {}
for name in data_dict:
    for feat in data_dict[name]:
        if data_dict[name][feat] == "NaN":
            if feat in nan_dict:
                nan_dict[feat] += 1
                # If more than 108 are NaN, I want to remove these features
                if nan_dict[feat] > 108:
                    remove_dict[feat] = nan_dict[feat]
            else:
                nan_dict[feat] = 1
#from pprint import pprint
#pprint(nan_dict)

# Remove features that have more than 75% NaN values
for feat in remove_dict:
    all_features.remove(feat)

### Determine which are the most important features using SelectKBest
from sklearn.feature_selection import SelectKBest
# Extract features and labels for SelectKBest
data = featureFormat(my_dataset, all_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Use data to look for outliers with the helper function
#visualize_plot(data)

# Set k to the desired number of features
k = 8

# Apply SelectKBest
selector = SelectKBest(k=k)
selector.fit(features, labels)
# Join each feature with its corresponding score then sort
scores = zip(all_features[1:], selector.scores_)
scores = sorted(scores, key=lambda scores: scores[1], reverse=True)
#pprint(scores)
# Create a list of the top k features
top_features = dict(scores[:k]).keys()

# Compile my features list using the top features
features_list = ['poi'] + top_features
#print features_list

# Extract top features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# FEATURE SCALING: Normalize features for classifiers
# (only required for some but can be applied to all)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

### Try a variety of classifiers
# Import classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
#from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

# Initialize classifiers
clf_NB = GaussianNB()
clf_DT = tree.DecisionTreeClassifier(min_samples_split=5,criterion='entropy')
#clf_SVC = SVC()
clf_KN = KNeighborsClassifier()
clf_RF = RandomForestClassifier()
clf_AB = AdaBoostClassifier()

# Leverage tester.py to fit and test the classifiers
test_classifier(clf_NB, my_dataset, features_list)
#test_classifier(clf_DT, my_dataset, features_list)
#test_classifier(clf_SVC, my_dataset, features_list)
#test_classifier(clf_KN, my_dataset, features_list)
#test_classifier(clf_RF, my_dataset, features_list)
#test_classifier(clf_AB, my_dataset, features_list)

# Apply Grid Search to fine tune the parameters
from sklearn import grid_search

# Set the parameters for my two chosen classifiers
parameters_DT = {'min_samples_split':[2,5,10,15,20],
                 'criterion':('gini','entropy')}

# Initialize Grid Search
#clf_GS_DT = grid_search.GridSearchCV(clf_DT, parameters_DT)

# Test the Grid Search classifier
#test_classifier(clf_GS_DT, my_dataset, features_list)

# Use best_params_ and grid_scores_ to identify the optimum parameters
#print clf_GS_DT.grid_scores_
#print clf_GS_DT.best_params_

# Apply optimum classifier
clf = clf_NB

# Dump classifier, dataset, and features_list
dump_classifier_and_data(clf, my_dataset, features_list)
