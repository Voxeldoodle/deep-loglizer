import re
import pandas as pd
import numpy as np
from utils import decision, json_pretty_dump
from collections import OrderedDict, defaultdict
from torch.utils.data import Dataset


seed = 42
np.random.seed(seed)

time_range = 60
train_anomaly_ratio = 0

params = {
    "log_file": log_path,
    "time_range": time_range,
    "label_file":label_path,
    "test_ratio": 0.2,
    "random_sessions": True,  # shuffle sessions
    "train_anomaly_ratio": train_anomaly_ratio,
}

# data_dir = os.path.join(data_dir, data_name)
# os.makedirs(data_dir, exist_ok=True)

def normalize_hdfs(df, label_file_path):
    res_df = df
    res_df["time"] = pd.to_datetime(
        res_df.Date.astype(str).str.zfill(6)+
        res_df.Time.astype(str).str.zfill(6),
        format="%d%m%y%H%M%S")

    # assign labels
    label_data = pd.read_csv(label_file_path, engine="c", na_filter=False, memory_map=True)
    label_data["Label"] = label_data["Label"].map(lambda x: int(x == "Anomaly"))
    label_data_dict = dict(zip(label_data["BlockId"], label_data["Label"]))

    column_idx = {col: idx for idx, col in enumerate(res_df.columns)}
    label_column = []
    for idx, row in enumerate(res_df.values):

        blkId_list = re.findall(r"(blk_-?\d+)", row[column_idx["Content"]])
        blkId_set = set(blkId_list)
        label_column.append(0)
        for blk_Id in blkId_set:
            if label_data_dict[blk_Id]:
                label_column[-1] = 1
                break
    res_df["Label"] = label_column
    return res_df

def normalize_bgl(df):
    res_df["time"] = pd.to_datetime(
        res_df.Date.astype(str).str.zfill(6)+
        res_df.Time.astype(str).str.zfill(6),
        format="%d%m%y%H%M%S")

    res_df["Label"] = res_df["Label"].map(lambda x: x != "-").astype(int).values

    return res_df


def df_session_split(
    log_df,
    time_range,
    test_ratio,
    train_ratio=None,
    log_file_path=None,
    train_anomaly_ratio=1,
    random_sessions=True
):
    if log_df is not None:
        struct_log = log_df
    else:
        try:
            struct_log = pd.read_csv(log_file_path, engine="c", na_filter=False, memory_map=True)
        except Exception as e:
            print("Provide either a valid log_df or log_file_path")

    struct_log["seconds_since"] = (
        (struct_log["time"] - struct_log["time"][0]).dt.total_seconds().astype(int)
    )

    session_dict = OrderedDict()
    column_idx = {col: idx for idx, col in enumerate(struct_log.columns)}
    for idx, row in enumerate(struct_log.values):
        current = row[column_idx["seconds_since"]]
        if idx == 0:
            sessid = current
        elif current - sessid > time_range:
            sessid = current
        if sessid not in session_dict:
            session_dict[sessid] = defaultdict(list)
        session_dict[sessid]["templates"].append(row[column_idx["EventTemplate"]])
        session_dict[sessid]["label"].append(
            row[column_idx["Label"]]
        )  # labeling for each log

    session_idx = list(range(len(session_dict)))
    # split data
    if random_sessions:
        print("Using random partition.")
        np.random.shuffle(session_idx)

    session_ids = np.array(list(session_dict.keys()))

    if train_ratio is None:
        train_ratio = 1 - test_ratio
    train_lines = int(train_ratio * len(session_idx))
    test_lines = int(test_ratio * len(session_idx))

    session_idx_train = session_idx[0:train_lines]
    session_idx_test = session_idx[-test_lines:]

    session_id_train = session_ids[session_idx_train]
    session_id_test = session_ids[session_idx_test]

    print("Total # sessions: {}".format(len(session_ids)))

    session_train = {
        k: session_dict[k]
        for k in session_id_train
        if (sum(session_dict[k]["label"]) == 0)
        or (sum(session_dict[k]["label"]) > 0 and decision(train_anomaly_ratio))
    }
    session_test = {k: session_dict[k] for k in session_id_test}

    session_labels_train = [
        1 if sum(v["label"]) > 0 else 0 for _, v in session_train.items()
    ]
    session_labels_test = [
        1 if sum(v["label"]) > 0 else 0 for _, v in session_test.items()
    ]

    train_anomaly = 100 * sum(session_labels_train) / len(session_labels_train)
    test_anomaly = 100 * sum(session_labels_test) / len(session_labels_test)

    print("# train sessions: {} ({:.2f}%)".format(len(session_train), train_anomaly))
    print("# test sessions: {} ({:.2f}%)".format(len(session_test), test_anomaly))

    # with open(os.path.join(data_dir, "session_train.pkl"), "wb") as fw:
    #     pickle.dump(session_train, fw)
    # with open(os.path.join(data_dir, "session_test.pkl"), "wb") as fw:
    #     pickle.dump(session_test, fw)
    # json_pretty_dump(params, os.path.join(data_dir, "data_desc.json"))
    # print("Saved to {}".format(data_dir))
    return session_train, session_test

def load_sessions(data_dir):
    with open(os.path.join(data_dir, "data_desc.json"), "r") as fr:
        data_desc = json.load(fr)
    with open(os.path.join(data_dir, "session_train.pkl"), "rb") as fr:
        session_train = pickle.load(fr)
    with open(os.path.join(data_dir, "session_test.pkl"), "rb") as fr:
        session_test = pickle.load(fr)

    train_labels = [
        v["label"] if not isinstance(v["label"], list) else int(sum(v["label"]) > 0)
        for _, v in session_train.items()
    ]
    test_labels = [
        v["label"] if not isinstance(v["label"], list) else int(sum(v["label"]) > 0)
        for _, v in session_test.items()
    ]

    num_train = len(session_train)
    ratio_train = sum(train_labels) / num_train
    num_test = len(session_test)
    ratio_test = sum(test_labels) / num_test
    logging.info("Load from {}".format(data_dir))
    logging.info(json.dumps(data_desc, indent=4))
    logging.info(
        "# train sessions {} ({:.2f} anomalies)".format(num_train, ratio_train)
    )
    logging.info("# test sessions {} ({:.2f} anomalies)".format(num_test, ratio_test))
    return session_train, session_test