# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .GxyNet import KeyPointModel


def build_model():
    model = KeyPointModel()
    return model
