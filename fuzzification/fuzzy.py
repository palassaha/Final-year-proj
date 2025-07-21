import pandas as pd
import numpy as np
import skfuzzy as fuzz

df = pd.read_csv(r"dataset\top_features_selected.csv")
features = ["Jitter:DDP", "MDVP:APQ", "HNR", "PPE", "Shimmer_Ratio"]
summary_stats = df[features].describe().T[["min", "25%", "50%", "75%", "max"]]
fuzzy_df = pd.DataFrame()

for feature in features:
    stats = summary_stats.loc[feature]
    x = np.linspace(stats["min"], stats["max"], 1000)
    low = fuzz.trimf(x, [stats["min"], stats["min"], stats["50%"]])
    med = fuzz.trimf(x, [stats["25%"], stats["50%"], stats["75%"]])
    high = fuzz.trimf(x, [stats["50%"], stats["max"], stats["max"]])
    
    for idx, val in enumerate(df[feature]):
        fuzzy_df.loc[idx, f"{feature}_Low"] = round(fuzz.interp_membership(x, low, val), 5)
        fuzzy_df.loc[idx, f"{feature}_Med"] = round(fuzz.interp_membership(x, med, val), 5)
        fuzzy_df.loc[idx, f"{feature}_High"] = round(fuzz.interp_membership(x, high, val), 5)

fuzzy_df.to_csv(r"dataset\fuzzy_top_features.csv", index=False)
print("Fuzzified data saved as 'fuzzy_top_features.csv'")
