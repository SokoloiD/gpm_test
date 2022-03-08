# -*- coding: utf-8 -*-

"""
функции аугментации и преобразования  изображений и лейблов

"""


def convert_image_to_rgb(image):
    return image.convert("RGB")


class PlayerIdToTeamId:
    def __init__(self, map_dict: dict):
        self.team_names = map_dict["teams"]
        self.team_name_to_id = {team_name: idx for idx, team_name in enumerate(self.team_names)}
        self.player_label_to_team = map_dict["class_to_team"]

    def __call__(self, player_name: str):
        return self.team_name_to_id[self.player_label_to_team[player_name]]
