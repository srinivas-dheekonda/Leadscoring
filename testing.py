import pandas as pd
df = pd.read_csv("/Users/srinivasaraodheekonda/Desktop/Data Science/Lead Scoring/Testing_for_DVC.csv")

df.drop(['lead_id'], axis=1, inplace=True)
df.to_csv("Testing_for_DVC.csv", index=False)