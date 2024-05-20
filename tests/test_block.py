import pytest
from pprint import pprint
from copy import deepcopy

from signbleu.block import (
    dict_to_block,
    list_to_block,
)

from utils import catch


data = {
    'face1': [{'gloss': 'happy', 'start': 0.099, 'end': 0.3}],
    'face2': [{'gloss': '::excited::', 'start': 0.3, 'end': 0.8}],
    'left': [
        {'gloss': 'hello', 'start': 0.1, 'end': 0.7},
        {'gloss': 'are', 'start': 3.5, 'end': 4.1},
    ],
    'right': [
        {'gloss': 'there', 'start': 1.1, 'end': 1.7},
        {'gloss': 'you', 'start': 2.8, 'end': 4.1},
    ],
    'both': [
        {'gloss': 'friend', 'start': 1.95, 'end': 2.7},
    ],
}
channel_combine_map = {
    'face1': 'face',
    'face2': 'face',
    'face3': 'face',
    'shoulder1': 'shoulder',
    'shoulder2': 'shoulder',
}
two_hand_map = {
    'both': ['left', 'right'],
}
default_blocks = [
    {'both': None, 'face1': 'happy:', 'face2': None, 'left': None, 'right': None},
    {'both': None, 'face1': ':happy', 'face2': None, 'left': 'hello:', 'right': None},
    {'both': None, 'face1': None, 'face2': '###::excited::###:', 'left': ':hello', 'right': None},
    {'both': None, 'face1': None, 'face2': ':###::excited::###', 'left': None, 'right': None},
    {'both': None, 'face1': None, 'face2': None, 'left': None, 'right': 'there'},
    {'both': 'friend', 'face1': None, 'face2': None, 'left': None, 'right': None},
    {'both': None, 'face1': None, 'face2': None, 'left': None, 'right': 'you:'},
    {'both': None, 'face1': None, 'face2': None, 'left': 'are', 'right': ':you'}
]

class TestBlock:
    def test_default_behavior(self):
        blocks = dict_to_block(
            data=data,
            #offset_threshold=0.002,
            #primary_channels=('right', 'left'),
            #channel_keys=None,
            #start_key='start',
            #end_key='end',
            #gloss_key='gloss',
            #channel_combine_map=channel_combine_map,
            #raise_on_error=True,
            #two_hand_map=two_hand_map,
        )
        with catch():
            assert blocks == default_blocks

    def test_error(self):
        global data
        data = deepcopy(data)
        data['face1'].append({'gloss': 'time_error_1', 'start': 10.0, 'end': 12.0})
        data['face2'].append({'gloss': 'time_error_2', 'start': 10.0, 'end': 12.0})
        with pytest.raises(RuntimeError):
            dict_to_block(
                data=data,
                #offset_threshold=0.002,
                #primary_channels=('right', 'left'),
                #channel_keys=None,
                #start_key='start',
                #end_key='end',
                #gloss_key='gloss',
                channel_combine_map={'face1': 'face', 'face2': 'face'},
                #raise_on_error=True,
                #two_hand_map=two_hand_map,
            )
