#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
how to use:

python3 predict_team_rest_server.py \
    -l ../model_weights/team_clf.pth \
    -t ../data/preprocessed/team_membership.json


curl --request POST -F "file=@./data/original/images/203.png" localhost:8000/predict

>"{\"status\": 1, \"description\": \"ok\", \"predicted_class_no\": 5, \"predicted_class_name\": \"yellow\"}"

"""
import io
import json
import logging
import os
from argparse import ArgumentParser
from dataclasses import dataclass

import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from torch import nn
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose

from src.lib.augmentation_transform import convert_image_to_rgb
from src.lib.common_utils import print_args
from src.lib.team_classifier_model import FootballTeamClassifier


@dataclass
class AppConfig:
    image_transformer: Compose
    model: nn.Module
    class_dict: dict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="football team  classifier ")
app_config = AppConfig(None, None, None)

log = logging.getLogger(__name__)


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("-l", "--load_model_file", dest="load_model_file", default="best_model.pth",
                        help="saved model file name")

    parser.add_argument("-t", "--team_map", dest="team_map", required=True, type=str,
                        help="path to json id to team map file")

    return parser.parse_args()


args = parse_arguments()


@app.on_event("startup")
async def set_up():
    print_args(args, log, "Start application with params ")

    with open(args.team_map, "rt") as f_in:
        name_to_team_dict = json.load(f_in)
    app_config.class_dict = name_to_team_dict["teams"]

    app_config.model = FootballTeamClassifier(classes_num=len(name_to_team_dict["teams"]))
    app_config.model.load_state_dict(torch.load(args.load_model_file, map_location=torch.device('cpu')))
    app_config.model.eval()
    app_config.image_transformer = Compose([Resize(299, max_size=300),
                                            CenterCrop([300, 150]),
                                            convert_image_to_rgb,
                                            ToTensor(),
                                            Normalize((0.48145466, 0.4578275, 0.40821073),
                                                      (0.26862954, 0.26130258, 0.27577711))
                                            ])


@app.get("/status")
def get_status() -> str:
    global app_config

    state = f"   Model is loaded:  {app_config.model is not None}" \
            f"   Transformer is loaded: {app_config.image_transformer is not None} " \
            f"   Map is loaded: {app_config.class_dict is not None}"
    return state


@app.get("/")
def root() -> str:
    return "Приложение для определения типа футбольной команды по изображению игрока"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    try:
        picture_stream = io.BytesIO(content)
        picture = Image.open(picture_stream)
        image_to_model = app_config.image_transformer(picture).unsqueeze(0)
        with torch.no_grad():
            predict = app_config.model(image_to_model).argmax(dim=1).squeeze(dim=0).item()

        result = {"status": 1,
                  "description": "ok",
                  "predicted_class_no": predict,
                  "predicted_class_name": app_config.class_dict[predict],
                  }
    except Exception as ex:
        result = {"status": 0,
                  "description": f"Exception : {ex}",
                  "predicted_class_no": -1,
                  "predicted_class_name": "None",
                  }

    return json.dumps(result, ensure_ascii=False).encode('utf8')


if __name__ == "__main__":
    uvicorn.run("predict_team_rest_server:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
