#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""

predict team by cropped image
usage :
python src/predict_team_cli.py \
    -l model_weights/team_clf.pth \
    -t data/preprocessed/team_membership.json \
    -i data/original/images/203.png


#Predicted class # 5  team name "yellow"


"""
import json
import logging
from argparse import ArgumentParser

import torch
from PIL import Image
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose

from lib.augmentation_transform import convert_image_to_rgb
from lib.common_utils import print_args
from lib.team_classifier_model import FootballTeamClassifier


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("-l", "--load_model_file", dest="load_model_file", default="best_model.pth",
                        help="saved model file name")

    parser.add_argument("-i", "--image_file", dest="image_file", required=True, type=str,
                        help="path to an images dir")

    parser.add_argument("-t", "--team_map", dest="team_map", required=True, type=str,
                        help="path to json id to team map file")

    return parser.parse_args()


def main(args):
    print_args(args, logger, "start train with params")
    with open(args.team_map, "rt") as f_in:
        name_to_team_dict = json.load(f_in)

    transforms = Compose([Resize(299, max_size=300),
                          CenterCrop([300, 150]),
                          convert_image_to_rgb,
                          ToTensor(),
                          Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                          ])

    image = Image.open(args.image_file)
    image_to_model = transforms(image).unsqueeze(0)

    model = FootballTeamClassifier(classes_num=len(name_to_team_dict["teams"]))
    model.load_state_dict(torch.load(args.load_model_file))
    model.eval()
    with torch.no_grad():
        predict = model(image_to_model).argmax(dim=1).squeeze(dim=0).cpu().item()
    print(f" Predicted class # {predict}  team name \"{name_to_team_dict['teams'][predict]}\"")


if __name__ == "__main__":
    main(parse_arguments())
