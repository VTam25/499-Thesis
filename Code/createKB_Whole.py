import pandas as pd
import numpy as np
import os

dataset_path = "./CSVs/KB_Whole.csv"
files_dir = "./CICS_Web_Parsed"
files = os.listdir(files_dir)

df = pd.read_csv(dataset_path, header=None)
df_numpy = np.array(df)
contents = []
for f in files:
    with open(files_dir+'/'+f, "r", encoding='utf-8') as input_page:
        contents.append(input_page.read())

df_numpy[1:, 2] = [text for text in contents]
final = pd.DataFrame(df_numpy)
print(final)
final.to_csv("./CSVs/KB_Whole_TEST.csv", header=False, index=False)