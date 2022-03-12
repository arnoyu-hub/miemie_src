import pickle as pkl
import pandas as pd
with open("SVM_dataset_sdg.pkl", "rb") as f:
    object = pkl.load(f)
    
df = pd.DataFrame(object)
df.to_csv(r'datasetH.csv')