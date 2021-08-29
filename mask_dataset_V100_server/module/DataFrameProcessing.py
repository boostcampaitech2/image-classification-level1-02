import os
import pandas as pd



def get_total_label_data_frame(df)(df):

    df["AgeBand"] = pd.cut(
        df["age"],
        bins = [df["age"].min(), 30, 60, 10000],
        right = False,
        labels = ["[0,30)","[30,60)" , "[60,inf)"]
    )
    new_df = {
        "Gender" : [],
        "AgeBand"   : [],
        "MaskState"  : [],
        "FileName"  : [],
    }
    for G, A, p in df[["gender", "AgeBand", "path"]].to_numpy():
        path = im_path + p + "/"
        for im_name in os.listdir(path):
            if im_name.startswith("."):
                continue
            elif im_name.startswith("mask"):
                M = "mask" # mask1,2,3,4,5
            elif im_name.startswith("normal"):
                M = "normal" # normal
            elif im_name.startswith("incorrect"):
                M = "incorrect" # incorrect_mask

            new_df["Gender"].append(G)
            new_df["AgeBand"].append(A)
            new_df["MaskState"].append(M)
            new_df["FileName"].append(p + "/" + im_name)
    
    return pd.DataFrame(new_df)