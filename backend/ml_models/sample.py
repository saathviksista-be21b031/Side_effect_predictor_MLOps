import pandas as pd

df=pd.read_csv("backend/data/FVA_Bounds/fva_bounds_DB00197.csv",header=None)
pred_df = pd.concat([df.iloc[:, 0], df.iloc[:, 1]], axis=0, ignore_index=True).to_frame()
print(pred_df)
print(pred_df.shape)