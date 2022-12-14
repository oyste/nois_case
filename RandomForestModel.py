import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

## PIPELINE FUNCTIONS
def target_encoder_dict(data, feature):
    categories = list(set(list(data[feature])))
    average_of_category = {}
    for cat in categories:
        average_of_category[cat] = data[data[feature] == cat]['SalePrice'].mean()
    return average_of_category

def price_target_feature(feature, xy_train, x_test):
    dict = target_encoder_dict(xy_train, feature)
    #print(xy_train[feature].isnull().sum())
    target_train = xy_train[feature].map(dict)
    target_test = x_test[feature].map(dict)
    if target_test.isna().sum() > 0:
        filler = target_test[target_test.isna()].mean()
        print(filler)
        target_test = target_test.fillna(0)
    print(target_test.isna().sum())
    return target_train, target_test, dict

def construct_features(xy_train, x_test, feats):
    xy_train_fix = xy_train.fillna('missing')
    x_test_fix = x_test.fillna('missing')
    train_feats, test_feats = [], []
    target_encoder_dicts = []
    for feat in feats:
        train, test, dict = price_target_feature(feat, xy_train_fix, x_test_fix)
        train_feats.append(train)
        test_feats.append(test)
        target_encoder_dicts.append(dict)
    pd.to_pickle(pd.concat(train_feats, axis=1, join='inner'), 'train_feats.pkl')
    train_out = np.hstack([e.values.reshape(-1,1) for e in train_feats])
    test_out = np.hstack([e.values.reshape(-1,1) for e in test_feats])
    return train_out, test_out, target_encoder_dicts


## LOAD DATA
df = pd.read_csv('Bulldozer Price Prediction Data/TrainAndValid.csv')

## FIX ALL DATATYPES TO BE CATEGORICAL (OBJECT) / PREPROCESSING
df.datasource = df.datasource.astype('object')
df.ModelID = df.ModelID.astype('object')
df.auctioneerID = df.auctioneerID.astype('object')
df.YearMade = df.YearMade.astype('object')
df.SalePrice = df.SalePrice.transform(lambda x: np.log1p(x))
df.saledate = df.saledate.transform(lambda x: x.split(' ')[0].split('/')[-1])

#for feat in df.columns:
#    print(feat, df[feat].describe())

## SPLIT DATA INTO TRAIN:CV (90:10)
cols_in_model = ['ModelID', 'datasource', 'YearMade', 'ProductGroup', 'saledate', 'fiBaseModel', 'fiModelDesc', 'Enclosure', 'Hydraulics', 'auctioneerID', 'SalePrice']
cols_in_x = cols_in_model[:-1]
df_xy = df[cols_in_model]
print(df_xy.head(n=10))
df_x = df_xy.drop('SalePrice', axis=1)
df_y = pd.DataFrame()
df_y['SalePrice'] = df.SalePrice
print(df_y)
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.1, random_state=42)
xy_train = pd.concat([x_train, y_train], axis=1, join='inner')

## FEATURE EXTRACTOR
x_train_feats, x_test_feats, target_encoder_dicts = construct_features(xy_train, x_test, cols_in_x)
y_train, y_test = y_train.values.flatten(), y_test.values.flatten()
filename_encoder_dicts = 'target_encoder_dicts.pkl'
pickle.dump(target_encoder_dicts, open(filename_encoder_dicts, 'wb'))

## REGRESSION AND PREDICTION
rfr = RandomForestRegressor(n_estimators=50, min_samples_split=3)
rfr.fit(x_train_feats, y_train)
y_pred = rfr.predict(x_test_feats).flatten()

## PRINT RESULTS
logdiff = y_pred-y_test
RMSLE = np.sqrt(np.mean(np.square(logdiff)))
print('RMSLE:', RMSLE)
print('Average absolute price error:', np.average(abs(np.expm1(y_pred) - np.expm1(y_test))))
print('Max absolute price error:', np.max(abs(np.expm1(y_pred) - np.expm1(y_test))))

## DUMP MODEL
filename_model = 'rfr_price_predictor.pkl'
pickle.dump(rfr, open(filename_model, 'wb'))
filename_train_xy = 'train_xy.pkl'
xy_train.to_pickle(filename_train_xy)