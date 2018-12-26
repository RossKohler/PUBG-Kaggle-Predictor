from sklearn.model_selection import train_test_split
from sklearn import neural_network
from sklearn import  metrics
from sklearn import preprocessing
import pandas as pd
import numpy as np
import random
import gc

gc.enable()

data_test = "./data/test_V2.csv"
data_train = "./data/train_V2.csv"


csvrows = 1000

features = []

# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df



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
df_train = pd.read_csv(data_train,nrows=csvrows)
df_train = reduce_mem_usage(df_train)

df_train = df_train[df_train.maxPlace>1]

print("where df_train null:\n",np.where(pd.isnull(df_train)))

features= list(df_train.columns)

features = [e for e in features if e not in {"winPlacePerc",
    'Id',
    'groupId',
    'matchId',
    'matchType'}]


df_train.replace([np.inf, -np.inf], np.nan)

print("Looking for NA values in Dataframes...")
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


print("Preparing data for training...")
Y_train = df_train['winPlacePerc']
Y_train = np.array(Y_train, dtype=np.float64);


X = df_train.copy().drop(['winPlacePerc'],axis=1)




del df_train
gc.collect()

X_train = X.drop(['Id','groupId','matchId','matchType'],axis=1)
X_train = np.array(X_train, dtype=np.float64);
del X
gc.collect()



scaler = preprocessing.MinMaxScaler(feature_range=(-1,1),copy=False).fit(X_train)
scaler.transform(X_train)
#Get Y in range of [-1,1]
Y_train = (Y_train*2)-1



model = neural_network.MLPRegressor(verbose=True,alpha=0.5,hidden_layer_sizes=(30,30))

print("Begin training of NN MLP...")
model.fit(X_train,Y_train)

print("Regression Report: %s:" % (model.score(X_train,Y_train)))

del X_train
del Y_train

gc.collect()

print("Reading test data...")
df_test = pd.read_csv(data_test,nrows=csvrows)
df_test = reduce_mem_usage(df_test)

print("Looking for NA values in Dataframes...")
print("Number of rows with NA in DF_TEST:",len(df_test[df_test.isnull().any(axis=1)]))

print("Preparing test data for predictions...")
df_test = addGroupSize(df_test)
df_test = total_distance(df_test)
df_test = items(df_test)
df_test = calculateGroupMin(df_test)
df_test = calculateGroupMean(df_test)
df_test = calculateGroupMax(df_test)
df_test = matchSize(df_test)


print("Preparing Test Data for predictions...")
X_test = df_test.drop(['Id','groupId','matchId','matchType'],axis=1)
X_test = np.array(X_test, dtype=np.float64);


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