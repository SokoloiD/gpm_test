# -*- coding: utf-8 -*-

"""
dataset class


"""

import logging
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from PIL import Image

from torch.utils import data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FootballPlayersDataset(data.Dataset):
    def __init__(self, csv_file_name: str, images_dir: str, transforms=None):
        super(FootballPlayersDataset, self).__init__()
        self.images_dir = images_dir
        self.transforms = transforms
        logger.info(f"try to load {csv_file_name}")
        self.df_players = pd.read_csv(csv_file_name)
        logger.info(f" {len(self.df_players)} records loaded")

        self.class_num_to_name = self.df_players["class_"].unique()
        self.classes_cnt = len(self.class_num_to_name)
        self.class_name_to_num = {label: class_num for class_num, label in enumerate(self.class_num_to_name)}

    def get_train_test_indexes(self, ratio: float, seed: int = None):
        """
        возвращает индексы для трейн и тест датасетов такие, что каждый класс в них представлен в соотношении ratio
        индексы основаны на случайной выборке


        :param ratio: train / test ratio
        :param seed: np.random  seed
        :return: tuple(train_index_list, test_index_list

        """
        assert 0 < ratio < 1
        class_indexes = defaultdict(list)

        for idx, row in self.df_players.iterrows():  # да, через iloc быстрее
            class_indexes[row["class_"]].append(idx)

        train_indexes = []
        test_indexes = []

        if seed is not None:
            np.random.seed(seed)

        for player_indexes in class_indexes.values():
            permuted_indexes = np.random.permutation(player_indexes)
            train_range = int(ratio * len(permuted_indexes))
            train_indexes.extend(permuted_indexes[0:train_range])
            test_indexes.extend(permuted_indexes[train_range:])

        return train_indexes, test_indexes

    def __len__(self):
        return len(self.df_players)

    def __getitem__(self, idx):
        box_id = self.df_players.iloc[idx]["boxid"]
        class_name = self.df_players.iloc[idx]["class_"]
        class_num = self.class_name_to_num[class_name]

        image_fname = os.path.join(self.images_dir, f"{box_id}.png")
        image = Image.open(image_fname)
        sample = {"image": image,
  #                "class_num": class_num,
                  "class_name": class_name,
                  }

        if self.transforms is not None:
            for item_name, transform in self.transforms:
                sample[item_name] = transform(sample[item_name])

        return sample
