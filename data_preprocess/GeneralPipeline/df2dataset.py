import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

import time

from session_splitter import *
from dataset_composer import *

start = time.time()
hdfs_path = "C:/Users/Antonio/Desktop/Tesi Magistrale/Codice/Resources/HDFS_parallel_main_structured.csv"
label_path = "C:/Users/Antonio/Desktop/Tesi Magistrale/Codice/Resources/HDFS/anomaly-label.csv"
hdfs_df = pd.read_csv(hdfs_path, engine="c", na_filter=False, memory_map=True)

df = normalize_hdfs(hdfs_df, label_path)

session_train, session_test = df_session_split(df, 60,.2)

middle = time.time()
print(f"Splitting Complete {middle-start:.2f}s")

ext = FeatureExtractor(label_type="next_log",  # "none", "next_log", "anomaly"
        window_type="sliding",
        window_size=50,
        stride=5)

session_train = ext.fit_transform(session_train)
session_test = ext.transform(session_test, datatype="test")

dataset_train = log_dataset(session_train)
dataloader_train = DataLoader(
    dataset_train, batch_size=512, shuffle=True, pin_memory=True
)

dataset_test = log_dataset(session_test)
dataloader_test = DataLoader(
    dataset_test, batch_size=4096, shuffle=False, pin_memory=True
)

end = time.time()
print(f"Processing Complete {end-start:.2f}s")