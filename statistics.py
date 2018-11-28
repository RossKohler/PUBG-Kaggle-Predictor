import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


csvrows = 100000000

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

def items(df):
    df['items'] = df['heals']+df['boosts']
    return df

def players_in_team(df):
    teamSize = df.groupby(['groupId']).size().to_frame('players_in_team')
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

def calcCorrAddFeatures():
    df = df_train.copy()
    df = items(df)
    df = players_in_team(df)
    df = total_distance(df)
    df = headshotKills_over_kills(df)
    df = killPlace_over_maxPlace(df)
    df = walkDistance_over_heals(df)
    df = walkDistance_over_kills(df)
    df = teamwork(df)

    df = df[['items','players_in_team','total_distance','headshotKills_over_kills','killPlace_over_maxPlace','walkDistance_over_heals','walkDistance_over_kills','teamwork','winPlacePerc']]
    corr=df.corr()
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
    #calcCorrelationHeatMap()
    calcCorrAddFeatures()

main()