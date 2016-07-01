import pandas as pd
import pandas_profiling
import numpy as np


df=pd.read_csv("dfRunLog.csv") 

# Example: Constant variable
df['source'] = "foo"

# Example: Highly correlated variables
df['steps_cor'] = df['steps'] + np.random.normal(scale=5,size=(len(df)))

pfr = pandas_profiling.ProfileReport(df)
pfr.to_file("example.html")