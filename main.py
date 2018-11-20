import numpy as numpy
import pandas as pd

print("Starting PUBG Kaggle Predictor...")

print("Reading training set...")
df_train = pd.read_csv("./data/train_V2.csv")

df_test = pd.read_csv("./data/test_V2.csv")
print("Reading test set...")

print("Training size:", df_train.shape)
print("Testing size:",df_test.shape)


print(df_train.head())

