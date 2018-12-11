from sklearn.model_selection import train_test_split
from sklearn import neural_network
from sklearn import  metrics
from sklearn import preprocessing
import pandas as pd
import numpy as np
import random
import gc

gc.enable()


csvrows = 500

features = []

def addGroupSize(df):
    groupSize=df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
    if("group_size" not in features):
        features.append("group_size")
    return pd.merge(df,groupSize,how='left',on=['matchId','groupId'])
    

def total_distance(df):
    df['total_distance'] = df['rideDistance']+df['swimDistance']+df['walkDistance']
    if("total_distance" not in features):
        features.append("total_distance")
    return df

def items(df):
    df['items'] = df['heals']+df['boosts']
    if("items" not in features):
        features.append("items")
    return df

def matchSize(df):
    matchSizes = df.groupby(['matchId']).size().reset_index(name='match_size')
    newDf = df.merge(matchSizes,how="left",on=["matchId"])
    return newDf

def calculateGroupMin(df):
    agg = df.groupby(['matchId','groupId'])[features].agg('min')
    agg_rank = agg.groupby(['matchId'])[features].rank(pct=True).reset_index()
    df_out = df.merge(agg.reset_index(),suffixes=["","_min"],how="left",on=["matchId","groupId"])
    df_out = df_out.merge(agg_rank,suffixes=["","_min_rank"],how="left",on=["matchId","groupId"])
    return df_out

def calculateGroupMean(df):
    agg = df.groupby(['matchId','groupId'])[features].agg('mean')
    agg_rank = agg.groupby(['matchId'])[features].rank(pct=True).reset_index()
    df_out = df.merge(agg.reset_index(),suffixes=["","_mean"],how="left",on=["matchId","groupId"])
    df_out = df_out.merge(agg_rank,suffixes=["","_mean_rank"],how="left",on=["matchId","groupId"])
    return df_out

def calculateGroupMax(df):
    agg = df.groupby(['matchId','groupId'])[features].agg('max')
    agg_rank = agg.groupby(['matchId'])[features].rank(pct=True).reset_index()
    df_out = df.merge(agg.reset_index(),suffixes=["","_max"],how="left",on=["matchId","groupId"])
    df_out = df_out.merge(agg_rank,suffixes=["","_max_rank"],how="left",on=["matchId","groupId"])
    return df_out

def averageWinPlacePercForGroup(df): 
    meanWinPlace = df.groupby(['groupId','matchId'])['winPlacePerc'].agg('mean').reset_index()
    meanWinPlace.to_csv("mean.csv");

    df['winPlacePerc'] = meanWinPlace
    return df



print("Reading training data...")
df_train = pd.read_csv("./data/train_V2.csv",nrows=csvrows)

df_train = df_train[df_train.maxPlace>1]

print("where df_train null:\n",np.where(pd.isnull(df_train)))

features= list(df_train.columns)

features = [e for e in features if e not in {"winPlacePerc",
    'Id',
    'groupId',
    'matchId',
    'matchType'}]


print("Reading test data...")
df_test = pd.read_csv("./data/test_V2.csv",nrows=csvrows)

df_train.replace([np.inf, -np.inf], np.nan)

print("Looking for NA values in Dataframes...")
print("Number of rows with NA in DF_TEST:",len(df_test[df_test.isnull().any(axis=1)]))
print("Number of rows with NA in DF_TRAIN:",len(df_train[df_train.isnull().any(axis=1)]))


df_train.dropna(inplace=True)


print("Adding additional Features...")
df_train = addGroupSize(df_train)

df_train = total_distance(df_train)
df_train = items(df_train)
df_train = calculateGroupMin(df_train)
df_train = calculateGroupMean(df_train)
df_train = calculateGroupMax(df_train)
df_train = matchSize(df_train)

#df_train = averageWinPlacePercForGroup(df_train)

df_test = addGroupSize(df_test)
df_test = total_distance(df_test)
df_test = items(df_test)
df_test = calculateGroupMin(df_test)
df_test = calculateGroupMean(df_test)
df_test = calculateGroupMax(df_test)
df_test = matchSize(df_test)

print("Preparing data for training...")
Y_train = df_train['winPlacePerc']
X = df_train.copy().drop(['winPlacePerc'],axis=1)




del df_train
gc.collect()

X_train = X.drop(['Id','groupId','matchId','matchType'],axis=1)

del X
gc.collect()


print("Number of features: %s"%len(X_train.columns))


scaler = preprocessing.MinMaxScaler(feature_range=(-1,1),copy=False).fit(X_train)
scaler.transform(X_train)
#Get Y in range of [-1,1]
Y_train = (Y_train*2)-1



model = neural_network.MLPRegressor(verbose=True,alpha=0.5,hidden_layer_sizes=(30,30))

print("Begin training of NN MLP...")
model.fit(X_train,Y_train)

print("Regression Report: %s:" % (model.score(X_train,Y_train)))

print("Preparing Test Data for predictions...")
X_test = df_test.drop(['Id','groupId','matchId','matchType'],axis=1)
scaler.transform(X_test)

print("Making Predictions...")

pred = model.predict(X_test)

del X_test
gc.collect()


print("Applying post processing...")
pred = (pred+1)/2

df_test['winPlacePerc'] = pred

del pred
gc.collect()

df_test.loc[df_test.winPlacePerc<0,"winPlacePerc"] = 0
df_test.loc[df_test.winPlacePerc>1,"winPlacePerc"] = 1

df_test.loc[df_test.maxPlace == 0, "winPlacePerc"] = 0
df_test.loc[df_test.maxPlace == 1, "winPlacePerc"] = 1

maxPlaceGreaterOne = df_test.loc[df_test.maxPlace > 1]

step = 1.0/(maxPlaceGreaterOne.maxPlace.values - 1)
adjustedPerc = np.around(maxPlaceGreaterOne.winPlacePerc.values/step)*step

df_test.loc[df_test.maxPlace > 1, "winPlacePerc"] = adjustedPerc

df_test.loc[(df_test.maxPlace > 1) & (df_test.numGroups == 1), "winPlacePerc"] = 0


submission = df_test[['Id', 'winPlacePerc']]
submission.to_csv('submission.csv', index=False)