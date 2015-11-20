#!/usr/bin/python

__owner__ = 'Shekhar Singh'

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import operator
from sklearn.cross_validation import train_test_split
import matplotlib
matplotlib.use("Pdf")
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict

class XGBoost:

	xgbModel = None
	xgb_params = {
		"objective": "reg:linear",
		"booster" : "gbtree",	# could use gblinear
		"eta": 0.09,	# 0.02, 0.06, 0.01
		"max_depth": 10,	# 7, 8
		"subsample": 0.9,	# 0.7
		"colsample_bytree": 0.9, #0.7, 1
		"silent": 1,
		"seed": 1000
	}
	xgb_num_boost_round = 1100	#1700	# nrounds
	fmap_path = 'data/xgb.fmap'
	
	def __init__(self):
		pass

	def trainXGBModel(self, train, features):
		print("Training an XGB Model..")
		# Reference: http://xgboost.readthedocs.org/en/latest/python/python_intro.html
		X_train, X_valid = train_test_split(train, test_size=0.012, random_state=10)
		dmtrain = xgb.DMatrix(X_train[features], np.log1p(X_train.Purchase))
		dmvalid = xgb.DMatrix(X_valid[features], np.log1p(X_valid.Purchase))
		watchlist = [(dmtrain, 'train'), (dmvalid, 'eval')]
		self.xgbModel = xgb.train( \
							self.xgb_params, \
							dmtrain, #dtrain \
							self.xgb_num_boost_round, #9857 \
							evals=watchlist, \
							#early_stopping_rounds=100, \
							feval=self.rmspe_xg, \
							verbose_eval=True \
						)
		return {'X_train' : X_train, 'X_valid' : X_valid, 'xgbModel' : self.xgbModel}

	def validateXGBModel(self, model_data):
		print("Validating the XGB Model..")
		yhat = model_data['xgbModel'].predict(xgb.DMatrix(model_data['X_valid'][self.features]))
		error = self.rmspe(model_data['X_valid'].Purchase.values, np.expm1(yhat))
		print('RMSPE: {:.6f}'.format(error))

	def predictUsingXGB(self, dataset, features):
		print("Predicting using XGB on the dataset..")
		return self.xgbModel.predict(xgb.DMatrix(dataset[features]))

	def calculateXGBFeatureImportances(self, features):
		self.generate_feature_map(features)
		importance = self.xgbModel.get_fscore(fmap=self.fmap_path)
		importance = sorted(importance.items(), key=operator.itemgetter(1))
		return importance

	def generate_feature_map(self, features):
		fp = open(self.fmap_path, 'w')
		for i, feat in enumerate(features):
		    fp.write('{0}\t{1}\tq\n'.format(i, feat))
		fp.close()
		print("XGB Map created at: " + self.fmap_path)

	def rmspe_xg(self, yhat, y):
		y = np.expm1(y.get_label())
		yhat = np.expm1(yhat)
		return "rmspe", self.rmspe(y,yhat)

	def rmspe(self, y, yhat):
		return np.sqrt(np.mean((yhat/y-1) ** 2))


class Problem(XGBoost):
  train = None
  test = None
  store = None
  types = None
  features = []
  datasets = {}

  def __init__(self, datasets):
    self.datasets = datasets
    self.types = {
      	# 'CompetitionOpenSinceYear' : np.dtype(int),
      	# 'CompetitionOpenSinceMonth' : np.dtype(int),
        'Gender' : np.dtype(str),
        'Occupation' : np.dtype(str),
        # 'SchoolHoliday' : np.dtype(float),
        # 'PromoInterval' : np.dtype(str)
    }

  def loadDataSets(self):
    print("Loading datasets(train|test|store)..")
    self.train = pd.read_csv(self.datasets['train'], dtype=self.types)
    self.test = pd.read_csv(self.datasets['test'], dtype=self.types)

  def dataFilterAndCleaning(self):
    #print("if store['Open'] is None, set 1..")
    self.train.fillna(999999, inplace=True)
    self.test.fillna(999999, inplace=True)

    # print("if Store is Closed, Ignore!")
    # self.train = self.train[self.train["Open"] != 0]
    # print("if Sales < 0, Ignore!")
    # self.train = self.train[self.train["Sales"] > 0]

    # print("Perform a join with store dataset..")
    # self.train = pd.merge(self.train, self.store, on='Store')
    # self.test = pd.merge(self.test, self.store, on='Store')

  def develop_features(self):
    print("Developing features..")
    self.generate_features(self.features, self.train)
    self.generate_features([], self.test)
    #self.features = list(OrderedDict.fromkeys(self.features))
    print(self.features)
    print('Features have been generated & training data processed.')

  def submission(self, test_probs):
    print("Saving the file at: "+self.datasets['submission_file'])
    result = pd.DataFrame({'Purchase': np.expm1(test_probs), "User_ID": self.test["User_ID"], "Product_ID": self.test["Product_ID"]})
    #result = result.sort_values(by='User_ID', ascending=True)    
    sample = pd.read_csv('data/sample_submission.csv')
    sample.User_ID = result['User_ID']
    sample.Product_ID = result['Product_ID']
    sample.Purchase = result['Purchase']
    sample.to_csv('output/submission.csv', index=False)
    #result.to_csv(self.datasets['submission_file'], index=False, cols=["User_ID","Product_ID","Purchase"])

  def plotting(self, importance, output='plot.png'):
    print("Plotting & Saving the file at: " + output)
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    # Kind: bar, barh, area
    featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(8, 12), color='Gray')
    plt.title('XGB Feature Importance')
    plt.ylabel('Features')
    plt.xlabel('Relative importance')
    fig_featp = featp.get_figure()
    fig_featp.savefig(output, bbox_inches='tight', pad_inches=0.5)

  def generate_features(self, features, data):
    data.fillna(999999, inplace=True)
	
    features.append('Product_ID_temp')
    data['Product_ID_temp']=data['Product_ID'].str[1:].astype(int).copy()

    features.extend(['Gender'])
    data.Gender.replace({'M':1, 'F':2}, inplace=True)

    features.extend(['Age'])
    age_mappings = {'0-17':1, '18-25':2, '26-35':3, '36-45':4, '46-50':5, '51-55':6, '55+':7}
    data.Age.replace(age_mappings, inplace=True)

    features.append('Occupation')
    data['Occupation'] = data['Occupation'].astype(int)

    features.extend(['City_Category'])
    data.City_Category.replace({'A':1, 'B':2, 'C':3}, inplace=True)

    features.extend(['Stay_In_Current_City_Years'])
    mappings = {'0':0, '1':1, '2':2, '3':3, '4':4, '4+':5}
    data.Stay_In_Current_City_Years.replace(mappings, inplace=True)

    features.append('Marital_Status')
    data['Marital_Status'] = data['Marital_Status'].astype(int)

    features.append('Product_Category_1')
    data['Product_Category_1'] = data['Product_Category_1'].astype(int)

    features.append('Product_Category_2')
    data['Product_Category_2'] = data['Product_Category_2'].astype(int)

    features.append('Product_Category_3')
    data['Product_Category_3'] = data['Product_Category_3'].astype(int)


if __name__ == '__main__':
	core_files = {
		'train' : "data/train.csv",
		'test' : "data/test.csv",
		'submission_file' : "output/submission.csv"
	}
	problem = Problem(core_files)
	problem.loadDataSets()
	problem.dataFilterAndCleaning()

	# Preparing to train data
	problem.develop_features()
	unique_features = []
	for feature in problem.features:
	    if feature not in unique_features:
	        unique_features.append(feature)
	problem.features = unique_features
	problem.xgb_num_boost_round = 9000
	model_data = problem.trainXGBModel(problem.train, problem.features)

	# Validating and Printing
	problem.validateXGBModel(model_data)
	test_probs = problem.predictUsingXGB(problem.test, problem.features)

	# # Submitting
	problem.submission(test_probs)

	# # Calculating & plotting Feature Importance
	problem.fmap_path = 'output/xgb.fmap'
	imp = problem.calculateXGBFeatureImportances(problem.features)
	problem.plotting(imp, output='output/plot_feature_importance_using_xgb.png')