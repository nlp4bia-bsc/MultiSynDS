import os
import pandas as pd

def extract_txt(path, filename):
    total_path = os.path.join(path, filename)
    return filename.split(".")[0], open(total_path, "r").read()

def files_to_df(path, extensions=["txt"]):
    files = [x for x in os.listdir(path) if x.split(".")[-1] in extensions]
    data = [extract_txt(path, f) for f in files]
    return pd.DataFrame(data, columns=["filenameid", "text"])
