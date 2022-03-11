#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
example:
Train team classifier model

train_clf.py  -s ../model_weights/team_clf.pth \
        -e 10 \
        -d ../data/original/images_labelling.csv \
        --images_dir ../data/original/images \
        -t ../data/preprocessed/team_membership.json

"""
import json
import logging
from argparse import ArgumentParser

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip, InterpolationMode

from lib.augmentation_transform import convert_image_to_rgb, PlayerIdToTeamId
from lib.common_utils import print_args
from lib.football_players_dataset import FootballPlayersDataset
from lib.team_classifier_model import FootballTeamClassifier
from lib.train_test_utils import train, validate, predict_from_dataloader

BICUBIC = InterpolationMode.BICUBIC
DEFAULT_TRAIN_TEST_RATIO = 0.8
LR = 1e-03
BATCH_SIZE = 32
DEFAULT_EPOCH = 10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("-s", "--save_model_name", dest="save_model_name", default="best_model.pth",
                        help="save model file name")

    parser.add_argument("-e", "--epochs", dest="epochs", default=DEFAULT_EPOCH, type=int,
                        help="path to an images_labeling file  ")
    parser.add_argument("-d", "--dataset", dest="dataset", required=True, type=str,
                        help="path to an images_labeling file  ")

    parser.add_argument("--images_dir", dest="images_dir", required=True, type=str,
                        help="path to an images dir")

    parser.add_argument("-t", "--team_map", dest="team_map", required=True, type=str,
                        help="path to json id to team map file")
    parser.add_argument("-r", "--train_test_ratio", dest="train_test_ratio", default=DEFAULT_TRAIN_TEST_RATIO,
                        type=float,
                        help="path to an images_labeling file  ")

    return parser.parse_args()


def main(args):
    print_args(args, logger, "start train with params")
    with open(args.team_map, "rt") as f_in:
        name_to_team_dict = json.load(f_in)
    n_px = 520
    transforms = [
        ("image", RandomHorizontalFlip()),
        ("image", Resize(299, max_size=300)),
        ("image", CenterCrop([300, 150])),
        # ("image", RandomPerspective(distortion_scale=0.1, p=1.0)),

        ("image", convert_image_to_rgb),
        ("image", ToTensor()),
        ("image", Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))),
        ("class_name", PlayerIdToTeamId(name_to_team_dict)),

    ]

    players_dataset = FootballPlayersDataset(args.dataset, args.images_dir, transforms=transforms)

    train_indexes, test_indexes = players_dataset.get_train_test_indexes(ratio=args.train_test_ratio, seed=0)

    train_dataset = Subset(players_dataset, train_indexes)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)

    test_dataset = Subset(players_dataset, test_indexes)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=4,
                                 pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device is: {device}")

    model = FootballTeamClassifier(classes_num=len(name_to_team_dict["teams"])).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2,
                                                           threshold=1e-5, verbose=True)

    best_val_loss = np.inf
    for epoch in range(args.epochs):
        train(model, loss_fn, optimizer, train_dataloader, device)
        val_loss = validate(model, loss_fn, test_dataloader, device)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if args.save_model_name is not None:
                logger.info(f"save model  {args.save_model_name}. Validation loss : {best_val_loss}")
                torch.save(model.state_dict(), args.save_model_name)

    logger.info(f"Finished. Best validation loss is :{best_val_loss}")

    model.load_state_dict(torch.load(args.save_model_name))
    ground_truth_labels, predict_labels = predict_from_dataloader(model, test_dataloader, device)

    conf_matr = confusion_matrix(ground_truth_labels, predict_labels)
    cls_rep = classification_report(ground_truth_labels, predict_labels)
    print("\n\nconfusion matrix")
    print(conf_matr)
    print("\n\nclassification report")
    print(cls_rep)


if __name__ == "__main__":
    main(parse_arguments())
