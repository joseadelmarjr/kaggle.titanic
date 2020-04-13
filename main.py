import pandas as pd

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_gs   = pd.read_csv("gender_submission.csv")

print("Linhas x colunas - gender_submission")
print(df_gs.shape)

print("########################")
print("Linhas x colunas - train")
print(df_train.shape)

print("########################")
print("Linhas x colunas - test")
print(df_test.shape)
