import pandas as pd

df = pd.read_csv("randomforest_sj.csv",header=2)

ensemble = df["Unnamed: 0"]
mode = df["Unnamed: 1"]


out = ensemble

for i in range(0,len(ensemble)-13):
    if(mode[i+13]>60):
        out[i] = max(ensemble[i],mode[i+13])
        
    if(mode[i]<6):
        out[i] = min(ensemble[i],mode[i])
    
df["res"] = out
df.to_csv("res.csv")
