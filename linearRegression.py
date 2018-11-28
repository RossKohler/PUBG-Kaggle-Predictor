import numpy as numpy
import pandas as pd
from sklearn import linear_model
import numpy as np


from sklearn.metrics import mean_squared_error,r2_score

csvrows = 10000000000

def items(df):
    df['items'] = df['heals']+df['boosts']
    return df

def players_in_team(df):
    teamSize = df.groupby(['groupId']).size().to_frame('players_in_team')

    print("teamSize:\n",teamSize.head())


    return df.merge(teamSize,how="left",on=['groupId'])

def total_distance(df):
    df['total_distance'] = df['rideDistance']+df['swimDistance']+df['walkDistance']
    return df

def headshotKills_over_kills(df):
    df['headshotKills_over_kills'] = df['headshotKills']/df['kills']
    df['headshotKills_over_kills'].fillna(0,inplace=True)
    return df

def killPlace_over_maxPlace(df):
    df['killPlace_over_maxPlace'] = df['killPlace']/df['maxPlace']
    df['killPlace_over_maxPlace'].fillna(0,inplace=True)
    df['killPlace_over_maxPlace'].replace(np.inf,0,inplace=True)
    return df

def walkDistance_over_heals(df):
    df['walkDistance_over_heals'] = df['walkDistance']/df['heals']
    df['walkDistance_over_heals'].fillna(0,inplace=True)
    df['walkDistance_over_heals'].replace(np.inf,0,inplace=True)
    return df

def walkDistance_over_kills(df):
    df['walkDistance_over_kills']=df['walkDistance']/df['heals']
    df['walkDistance_over_kills'].fillna(0,inplace=True)
    df['walkDistance_over_kills'].replace(np.inf,0,inplace=True)
    return df

def teamwork(df):
    df['teamwork'] = df['assists'] + df['revives']
    return df




def addAdditionalFeatures(df):
        df = items(df)
        df = players_in_team(df)
        df = total_distance(df)
        df = headshotKills_over_kills(df)
        df = killPlace_over_maxPlace(df)
        df = walkDistance_over_heals(df)
        df = walkDistance_over_kills(df)
        df = teamwork(df)
        return df






df_train = pd.read_csv("./input/train_V2.csv",nrows=csvrows)
df_test = pd.read_csv("./input/test_V2.csv",nrows=csvrows)

print("Adding additional features...")

df_train = addAdditionalFeatures(df_train)
df_test = addAdditionalFeatures(df_test)

print("Training data....")
print(df_train.head())
print(df_train.columns)
print(len(df_train))

print("Test data...")
print(df_test.head())
print(df_test.columns)
print(len(df_test))

matchTypes = df_train.matchType.unique()

print("matchTypes:",matchTypes)


matchTypeModels={}

meanError = []
variance = []


for matchType in matchTypes:
    df_train_sub = df_train.loc[df_train["matchType"]==matchType]
    train_Id = df_train_sub["Id"]
    df_train_sub = df_train_sub.drop(['Id','groupId','matchId','matchType'],axis=1)
    df_train_sub = df_train_sub.fillna(df_train_sub.mean())
    X_train = df_train_sub.drop(['winPlacePerc'],axis=1)
    Y_train = df_train_sub['winPlacePerc']
    linearFit = linear_model.LinearRegression()
    matchTypeModels[matchType] = linearFit.fit(X_train,Y_train)
    y_train_pred = linearFit.predict(X_train)

    print("Printing performance metrics for: %s"%matchType)
    print("Coeficients: \n",linearFit.coef_)

    mse = mean_squared_error(y_train_pred,Y_train)
    var = r2_score(y_train_pred,Y_train)
    print("Variance: %.2f" %var)
    print("Mean Squared Error: %.2f" %mse)

    meanError.append(mean_squared_error(y_train_pred,Y_train))
    variance.append(var)

print("Total Mean Squared Error: %.2f" %(sum(meanError)/float(len(meanError))))
print("Average Variance: %.2f" %(sum(variance)/float(len(variance))))    





result = pd.DataFrame()


for matchType in matchTypes:

    df_test_sub = df_test.loc[df_test["matchType"]==matchType]

    test_Id = df_test_sub[['Id']]
    test_Id = test_Id.reset_index(drop=True)


    df_test_sub = df_test_sub.drop(['Id','groupId','matchId','matchType'],axis=1)
    X_test = df_test_sub.fillna(df_test_sub.mean())

    Y_pred = matchTypeModels[matchType].predict(X_test)
    y_pred = pd.DataFrame(Y_pred)
    sub_result = pd.concat([test_Id,y_pred],axis=1)
    result = result.append(sub_result)

result.columns=['Id','winPlacePerc']
print("Result Shape:",result.shape)
result.to_csv("submission.csv",index=False)