import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



csvrows = 1000

df_train = pd.DataFrame()
df_test = pd.DataFrame()


def readData():
    print("Reading training set...")
    global df_train
    global df_test


    df_train = pd.read_csv("./data/train_V2.csv",nrows=csvrows)
    df_train = df_train[df_train["winPlacePerc"].notnull()].reset_index(drop=True)

    df_train.loc[:,"winPlacePerc"]*=100

    df_test = pd.read_csv("./data/test_V2.csv",nrows=csvrows)
    print("Reading test set...")

    print("Training size:", df_train.shape)
    print("Testing size:",df_test.shape)
    print(df_train.head())


def plotKillHistorgram():
    print("Ploting historgam...")
    train_hist = df_train.hist(column="winPlacePerc",bins=100)
    plt.show(block=True)

def plotGameModeCount():
    print("Showing different game modes...")
    game_modes = df_train["matchType"].value_counts().sort_values(ascending=False).plot(kind="bar")
    plt.show(block=True)


def calcCorrelationHeatMap():
        print("Calcuating Correlation Heatmap..")
        drop_cols = ["Id","groupId","matchId","matchType"]
        print(df_train.columns)


        cols_to_fit = [col for col in df_train.columns if col not in drop_cols]
        print(cols_to_fit)
        corr = df_train[cols_to_fit].corr()
        sns.heatmap(
            data=corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            annot=True,
            linecolor='white',
            linewidths=0.1,
            cmap="RdBu"
        )
        plt.show(block=True)



def main():
    readData()
    #plotKillHistorgram()
    calcCorrelationHeatMap()

main()