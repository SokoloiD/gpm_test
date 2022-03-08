# -*- coding: utf-8 -*-

"""
common utils

"""


def print_args(args, logger, title: str):
    logger.info(title)
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)
