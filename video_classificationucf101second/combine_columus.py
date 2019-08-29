import pandas as pd
dfs = []
dfs.append( pd.read_csv("E:/Moe_Junyue/dataset/predictions_train_moe.csv",header=None) )
dfs.append( pd.read_csv("E:/Moe_Junyue/dataset/predictions_train_22.csv",header=None) )
dfs.append( pd.read_csv("E:/Moe_Junyue/dataset/predictions_dbof.csv",header=None) )
print(pd.concat(dfs,axis=1).to_csv("E:/Moe_Junyue/dataset/3combined.csv",header=False,index=False))