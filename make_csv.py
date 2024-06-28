# %%
import pandas as pd

df = pd.read_csv("/mnt/passive-acoustic-biodiversity/Peru_2019_Audiomoth_Sound_Recordings/2019_Peru_MDD_AudioMoth_Recordings_Metadata_Firmware_Timing_Error_Corrected_Faulty_Clips_Removed.csv")

# %%
df

# %%
df["CLIP LENGTH"] = df["Duration"]

# %%
import math 
def create_raw_chunks(row):
    row = row.iloc[0]
    rows = []
    for i in range(0, math.floor(row["CLIP LENGTH"]), 5):
        row_temp = row.copy(deep=True)
        row_temp["OFFSET"] = i
        row_temp["DURATION"] = 5
        rows.append(row_temp.to_frame().T)
    return pd.concat(rows)


chunked_df = df.groupby("SourceFile", as_index=False).apply(create_raw_chunks).reset_index()
chunked_df.to_csv("peru-2019-pyha-anaylzer-inferance.csv")


