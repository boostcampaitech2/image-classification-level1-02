import os
import pandas as pd



def get_total_label_data_frame(df,im_path):

    df["AgeBand"] = pd.cut(
        df["age"],
        bins = [df["age"].min(), 30, 60, 10000],
        right = False,
        labels = ["[0,30)","[30,60)" , "[60,inf)"]
    )
    new_df = {
        "Gender"    : [],
        "AgeBand"   : [],
        "MaskState" : [],
        "FileName"  : [],
        "Label"     : []
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
            new_df["Label"].append("_".join([G, A, M]))
    
    return pd.DataFrame(new_df)

def total_label_balance(df):
    '''
    Extract a 100 subsamples from each total_label.
    The minimum sample labal is male_[60,inf)_(mask,normal,incorrect)
    and its number of samples is 83.
    This function apply 100 subsampling for each labels except male_[60,inf)_(m,n,i)
    '''
    # ToDo : 마스크 라벨에 사람이 중복되게 들어가있을 가능성이 있다.
    subsample_list = []
    for label in df["Label"].unique():
        _df = df[df["Label"] == label]
        try:
            subsample_list.append(_df.sample(100))
        except:
            subsample_list.append(_df.sample(len(_df)))
    df = pd.concat(subsample_list, axis = 0)
    df = df.sample(len(df)) # suffle
    return df