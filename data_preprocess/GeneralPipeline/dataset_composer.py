import os
import io
import itertools
import torch
import numpy as np
from collections import Counter, defaultdict
from sklearn.base import BaseEstimator
import hashlib
import pickle
import re
import logging
from torch.utils.data import Dataset

from utils import json_pretty_dump, dump_pickle, load_pickle

class FeatureExtractor(BaseEstimator):
    """
    label_type: "none", "next_log", "anomaly"
    window_type: "session", "sliding"
    """

    def __init__(
        self,
        label_type="next_log",
        window_type="sliding",
        window_size=None,
        stride=None,
        **kwargs,
    ):
        self.label_type = label_type
        self.window_type = window_type
        self.window_size = window_size
        self.stride = stride

    def __generate_windows(self, session_dict, extra_metric=False):
        window_count = 0
        for session_id, data_dict in session_dict.items():
            if self.window_type == "sliding":
                i = 0
                templates = data_dict["templates"]
                template_len = len(templates)
                windows = []
                window_labels = []
                window_anomalies = []

                if extra_metric:
                    window_an_classes = []

                while i + self.window_size < template_len:
                    window = templates[i: i + self.window_size]
                    next_log = self.log2id_train.get(templates[i + self.window_size], 1)

                    if isinstance(data_dict["label"], list):
                        window_anomaly = int(
                            1 in data_dict["label"][i: i + self.window_size + 1]
                        )
                        if extra_metric:
                            classes = [self.log2id_train.get(templates[j], 1)
                            if data_dict["label"][j] > 0 else 0
                            for j in range(i, i + self.window_size )
                            ]
                    else:
                        window_anomaly = data_dict["label"]
                        if extra_metric:
                            classes = [self.log2id_train.get(templates[i], 1)*data_dict["label"]
                                for i in range(i, i + self.window_size )]

                    windows.append(window)
                    window_labels.append(next_log)
                    window_anomalies.append(window_anomaly)
                    if extra_metric:
                        window_an_classes.append(np.array(classes))

                    i += self.stride
                else:
                    window = templates[i:-1]
                    window.extend(["PADDING"] * (self.window_size - len(window)))
                    next_log = self.log2id_train.get(templates[-1], 1)

                    if isinstance(data_dict["label"], list):
                        window_anomaly = int(1 in data_dict["label"][i:])
                        if extra_metric:
                            classes = [self.log2id_train.get(templates[j], 1)
                            if data_dict["label"][j] > 0 else 0
                            for j in range(len(templates[i:]))
                            ]
                            classes.extend([1] * (self.window_size - len(classes)))
                    else:
                        window_anomaly = data_dict["label"]
                        if extra_metric:
                            classes = [self.log2id_train.get(templates[i], 1)*data_dict["label"]
                            for i in range(len(templates[i:]))]
                            classes.extend([1] * (self.window_size - len(classes)))

                    windows.append(window)
                    window_labels.append(next_log)
                    window_anomalies.append(window_anomaly)
                    if extra_metric:
                        window_an_classes.append(np.array(classes))
                window_count += len(windows)

                session_dict[session_id]["windows"] = windows
                session_dict[session_id]["window_labels"] = window_labels
                session_dict[session_id]["window_anomalies"] = window_anomalies

                if extra_metric:
                    session_dict[session_id]["window_an_classes"] = window_an_classes

                if session_id == "all":
                    logging.info(
                        "Total window number {} ({:.2f})".format(
                            len(window_anomalies),
                            sum(window_anomalies) / len(window_anomalies),
                        )
                    )

            elif self.window_type == "session":
                session_dict[session_id]["windows"] = [data_dict["templates"]]
                session_dict[session_id]["window_labels"] = [data_dict["label"]]
                window_count += 1

        logging.info("{} sliding windows generated.".format(window_count))

    def __windows2quantitative(self, windows):
        total_features = []
        for window in windows:
            feature = [0] * len(self.id2log_train)
            window = [self.log2id_train.get(x, 1) for x in window]
            log_counter = Counter(window)
            for logid, log_count in log_counter.items():
                feature[int(logid)] = log_count
            total_features.append(feature[1:])  # discard the position of padding
        return np.array(total_features)

    def __windows2sequential(self, windows):
        total_features = []
        for window in windows:
            ids = [self.log2id_train.get(x, 1) for x in window]
            total_features.append(ids)
        return np.array(total_features)

    def fit(self, session_dict):
        log_padding = "<pad>"
        log_oov = "<oov>"

        # encode
        total_logs = list(
            itertools.chain(*[v["templates"] for k, v in session_dict.items()])
        )
        self.ulog_train = set(total_logs)
        self.id2log_train = {0: log_padding, 1: log_oov}
        self.id2log_train.update(
            {idx: log for idx, log in enumerate(self.ulog_train, 2)}
        )
        self.log2id_train = {v: k for k, v in self.id2log_train.items()}

        logging.info("{} templates are found.".format(len(self.log2id_train)))

        if self.label_type == "next_log":
            print("num_labels ", len(self.log2id_train))
        elif self.label_type == "anomaly":
            print("num_labels ", 2)
        else:
            logging.info('Unrecognized label type "{}"'.format(self.label_type))
            exit()

        if self.feature_type == "sequentials":+
            print("vocab_size ", len(self.log2id_train))

        else:
            logging.info('Unrecognized feature type "{}"'.format(self.feature_type))
            exit()

    def transform(self, session_dict, datatype="train"):
        logging.info("Transforming {} data.".format(datatype))
        ulog = set(itertools.chain(*[v["templates"] for k, v in session_dict.items()]))
        if datatype == "test":
            # handle new logs
            ulog_new = ulog - self.ulog_train
            logging.info(f"{len(ulog_new)} new templates found while transforming.")
            self.id2log = self.id2log_train
            self.id2log.update(
                {idx: log for idx, log in enumerate(self.ulog_new, len(self.ulog_train))}
            )
            self.log2id = {v: k for k, v in self.id2log.items()}

        # generate windows, each window contains logid only
        if datatype == "train":
            self.__generate_windows(session_dict, self.stride)
        else:
            self.__generate_windows(session_dict, self.stride)

        for session_id, data_dict in session_dict.items():
            feature_dict = defaultdict(list)
            windows = data_dict["windows"]
            # generate sequential feautres # sliding windows on logid list
            if self.feature_type == "sequentials":
                feature_dict["sequentials"] = self.__windows2sequential(windows)

            # generate quantitative feautres # count logid in each window
            if self.feature_type == "quantitatives":
                feature_dict["quantitatives"] = self.__windows2quantitative(windows)

            session_dict[session_id]["features"] = feature_dict

        logging.info("Finish feature extraction ({}).".format(datatype))
        return session_dict

    def fit_transform(self, session_dict):
        self.fit(session_dict)
        return self.transform(session_dict, datatype="train")

class log_dataset(Dataset):
    def __init__(self, session_dict, feature_type="sequentials"):
        extra_metric = "window_an_classes" in next(iter(session_dict.values()))
        flatten_data_list = []
        # flatten all sessions
        for session_idx, data_dict in enumerate(session_dict.values()):
            features = data_dict["features"][feature_type]
            window_labels = data_dict["window_labels"]
            window_anomalies = data_dict["window_anomalies"]
            if extra_metric:
                window_an_classes = data_dict["window_an_classes"]
            for window_idx in range(len(window_labels)):
                sample = {
                    "session_idx": session_idx,  # not session id
                    "features": features[window_idx],
                    "window_labels": window_labels[window_idx],
                    "window_anomalies": window_anomalies[window_idx]
                    }
                if extra_metric:
                    sample["window_an_classes"] = window_an_classes[window_idx]
                flatten_data_list.append(sample)
        self.flatten_data_list = flatten_data_list

    def __len__(self):
        return len(self.flatten_data_list)

    def __getitem__(self, idx):
        return self.flatten_data_list[idx]
