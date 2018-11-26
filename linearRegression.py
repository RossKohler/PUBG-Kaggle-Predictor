import numpy as numpy
import pandas as pd
from sklearn import linear_model

from sklearn.metrics import mean_squared_error,r2_score




df_train = pd.read_csv("./data/train_V2.csv")
df_test = pd.read_csv("./data/test_V2.csv")

df_sample_submission = pd.read_csv("./data/sample_submission_V2.csv")


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
    df_train_sub = df_train_sub(['Id','groupId','matchId','matchType'],axis=1)
    df_train_sub = df_train_sub(df_train_sub.mean())
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
    test_Id = df_test_sub['Id']
    df_test_sub = df_test_sub.drop(['Id','groupId','matchId','matchType'],axis=1)
    X_test = df_test_sub.fillna(df_test.sub.mean())
    y_pred = matchTypeModels[matchType].prdict(X_test)
    y_pred = pd.DataFrame(y_pred)
    sub_result = pd.concat([test_Id,y_pred],axis=1)
    result = pd.concat([result,sub_result],axis=0)

result.columns=['Id','winPlacePerc']
print(result)
result.to_csv("./output/submission.csv",index=False)

"""
train_Id = df_train['Id']
df_train = df_train.drop(['Id','groupId','matchId','matchType'],axis=1)
df_train = df_train.fillna(df_train.mean())
X_train = df_train.drop(['winPlacePerc'],axis=1)
Y_train = df_train['winPlacePerc']

test_Id = df_test['Id']
df_test = df_test.drop(['Id','groupId','matchId','matchType'],axis=1)
X_test = df_test.fillna(df_test.mean())


linearFit = linear_model.LinearRegression()
linearFit.fit(X_train,Y_train)


Y_test = linearFit.predict(X_train)
Y_pred = linearFit.predict(X_test)

print("Coeficients: \n",linearFit.coef_)
print("Variance: %.2f" %r2_score(Y_test,Y_train))
print("Mean Squared Error: %.2f" %mean_squared_error(Y_test,Y_train))


y_pred = pd.DataFrame(Y_pred)
result = pd.concat([test_Id,y_pred],axis=1)
result.columns=['Id','winPlacePerc']
print(result)

result.to_csv("./output/submission.csv",index=False)

"""


























