from sklearn.model_selection import train_test_split
from sklearn import neural_network
from sklearn import  metrics
import pandas as pd


csvrows = 1000000000000

print("Reading training data...")
df_train = pd.read_csv("./data/train_V2.csv",nrows=csvrows)

print("Reading test data...")
df_test = pd.read_csv("./data/test_V2.csv",nrows=csvrows)

df_train = df_train.fillna(df_train.mean())
df_test= df_test.fillna(df_test.mean())



print("Preparing data for training...")
Y = df_train['winPlacePerc']
X= df_train.copy().drop(['winPlacePerc'],axis=1)

X = X.drop(['Id','groupId','matchId','matchType'],axis=1)

print("X size:",len(X),"Y size:",len(Y))
print("X head:",X.head(),"Y head:",Y.head())


x_train, x_val, y_train, y_val = train_test_split(X,Y,test_size=0.20,random_state=291)



model = neural_network.MLPRegressor(activation="relu",alpha=1e-5,hidden_layer_sizes=(100,),solver='lbfgs',random_state=123)

print("Begin training of NN MLP...")
model.fit(x_train,y_train)

predicted = model.predict(x_val)
print("Regression Report:\n %s:" % (model.score(x_val,y_val)))


