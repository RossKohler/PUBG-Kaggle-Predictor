import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt

import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import plotly.offline as py


print("Starting PUBG Kaggle Predictor...")

csvrows = 1000000000


print("Reading training set...")
df_train = pd.read_csv("./data/train_V2.csv",nrows=csvrows)
df_train = df_train[df_train["winPlacePerc"].notnull()].reset_index(drop=True)

df_train.loc[:,"winPlacePerc"]*=100


df_test = pd.read_csv("./data/test_V2.csv",nrows=csvrows)
print("Reading test set...")

print("Training size:", df_train.shape)
print("Testing size:",df_test.shape)


print(df_train.head())

print("Ploting historgam...")
train_hist = df_train.hist(column="winPlacePerc",bins=100)
plt.show(block=True)

print("Showing different game modes...")
game_modes = df_train["matchType"].value_counts().sort_values(ascending=False).plot(kind="bar")

plt.show(block=True)

