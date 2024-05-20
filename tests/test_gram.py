r"""
Test gram logic.

TODO:
- Update whitespace-related tests. Currently, whitespace for
  `original_y_grams=False` is being calculated incorrectly (some None values
  are being dropped).
"""

import pytest
from pprint import pprint
from copy import deepcopy

from signbleu.block import (
    dict_to_block,
)
from signbleu.gram import (
    Gram1D,
    Gram2D,
    block_to_gram,
    GramUtilities,
)
from signbleu.constants import SEP_KEY

from utils import catch


gram_utils = GramUtilities()
construct_gram = GramUtilities().construct_gram
data = {
    'face1': [{'gloss': 'happy', 'start': 0.099, 'end': 0.3}],
    'face2': [{'gloss': '::excited::', 'start': 0.3, 'end': 0.8}],
    'left': [
        {'gloss': 'hello', 'start': 0.1, 'end': 0.7},
        {'gloss': 'are', 'start': 3.5, 'end': 4.1},
    ],
    'right': [
        {'gloss': 'oh', 'start': 0.099, 'end': 0.8},
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
reduced_blocks = [  # blocks with both channel_combine_map and two_hand_map applied
    {'face': 'happy:', 'left': None, 'right': 'oh:'},
    {'face': ':happy', 'left': 'hello:', 'right': ':oh:'},
    {'face': '###::excited::###:', 'left': ':hello', 'right': ':oh:'},
    {'face': ':###::excited::###', 'left': None, 'right': ':oh'},
    {'face': None, 'left': None, 'right': 'there'},
    {'face': None, 'left': 'friend', 'right': 'friend'},
    {'face': None, 'left': None, 'right': 'you:'},
    {'face': None, 'left': 'are', 'right': ':you'}
]

default_2x1d_grams_original = {
    't1': [
        Gram1D(["face_happy/2"]),
        Gram1D(["face_###excited###/2"]),
        Gram1D(["left_hello/2"]),
        Gram1D(["left_friend/1"]),
        Gram1D(["left_are/1"]),
        Gram1D(["right_oh/4"]),
        Gram1D(["right_there/1"]),
        Gram1D(["right_friend/1"]),
        Gram1D(["right_you/2"]),
    ],
    't2': [
        Gram1D(["face_happy/2", "face_###excited###/2"]),
        Gram1D(["left_hello/2", "left_friend/1"]),
        Gram1D(["left_friend/1", "left_are/1"]),
        Gram1D(["right_oh/4", "right_there/1"]),
        Gram1D(["right_there/1", "right_friend/1"]),
        Gram1D(["right_friend/1", "right_you/2"]),
    ],
    'c2': [
        Gram1D(["face_happy", "right_oh"]),
        Gram1D(["face_happy", "left_hello"]),
        Gram1D(["face_happy", "right_oh"]),
        Gram1D(["left_hello", "right_oh"]),
        Gram1D(["face_###excited###", "left_hello"]),
        Gram1D(["face_###excited###", "right_oh"]),
        Gram1D(["left_hello", "right_oh"]),
        Gram1D(["face_###excited###", "right_oh"]),
        Gram1D(["left_friend", "right_friend"]),
        Gram1D(["left_are", "right_you"]),
    ],
    't3': [
        Gram1D(["left_hello/2", "left_friend/1", "left_are/1"]),
        Gram1D(["right_oh/4", "right_there/1", "right_friend/1"]),
        Gram1D(["right_there/1", "right_friend/1", "right_you/2"]),
    ],
}

default_2d_grams_whitespace = {
    't1': [
        Gram1D(["face_happy/2"]),
        Gram1D(["face_###excited###/2"]),
        Gram1D(["left_hello/2"]),
        Gram1D(["left_friend/1"]),
        Gram1D(["left_are/1"]),
        Gram1D(["right_there/1"]),
        Gram1D(["right_friend/1"]),
        Gram1D(["right_you/2"]),
    ],
    't2': [
        Gram1D(["face_happy/2", "face_###excited###/2"]),
        Gram1D(["face_###excited###/2", "face_None/4"]),
        Gram1D(["left_None/1", "left_hello/2"]),
        Gram1D(["left_hello/2", "left_None/2"]),
        Gram1D(["left_None/2", "left_friend/1"]),
        Gram1D(["left_friend/1", "left_None/1"]),
        Gram1D(["left_None/1", "left_are/1"]),
        Gram1D(["right_None/4", "right_there/1"]),
        Gram1D(["right_there/1", "right_friend/1"]),
        Gram1D(["right_friend/1", "right_you/2"]),
    ],
    'c2': [
        Gram1D(["face_happy", "left_hello"]),
        Gram1D(["face_###excited###", "left_hello"]),
        Gram1D(["face_###excited###", "left_None"]),
        Gram1D(["face_None", "left_friend"]),
        Gram1D(["face_None", "left_are"]),
        Gram1D(["face_happy", "right_None"]),
        Gram1D(["face_###excited###", "right_None"]),
        Gram1D(["face_None", "right_there"]),
        Gram1D(["face_None", "right_friend"]),
        Gram1D(["face_None", "right_you"]),
        Gram1D(["left_hello", "right_None"]),
        Gram1D(["left_None", "right_there"]),
        Gram1D(["left_friend", "right_friend"]),
        Gram1D(["left_are", "right_you"]),
    ],
    't3': [
        Gram1D(["face_happy/2", "face_###excited###/2", "face_None/4"]),
        Gram1D(["left_None/1", "left_hello/2", "left_None/2"]),
        Gram1D(["left_hello/2", "left_None/2", "left_friend/1"]),
        Gram1D(["left_None/2", "left_friend/1", "left_None/1"]),
        Gram1D(["left_friend/1", "left_None/1", "left_are/1"]),
        Gram1D(["right_None/4", "right_there/1", "right_friend/1"]),
        Gram1D(["right_there/1", "right_friend/1", "right_you/2"]),
    ],
    't2c2': [
        Gram2D([["face_happy/2"], ["left_None/1", "left_hello/1"]]),
        Gram2D([["face_happy/1", "face_###excited###/1"], ["left_hello/2"]]),
        Gram2D([["face_###excited###/2"], ["left_hello/1", "left_None/1"]]),
        Gram2D([["face_###excited###/1", "face_None/1"], ["right_None/1", "right_there/1"]]),
        Gram2D([["left_None/1", "left_friend/1"], ["right_there/1", "right_friend/1"]]),
        Gram2D([["left_friend/1", "left_None/1"], ["right_friend/1", "right_you/1"]]),
        Gram2D([["left_None/1", "left_are/1"], ["right_you/2"]]),
    ]
}


default_2d_grams_nowhitespace = {
    't1': [
        Gram1D(["face_happy/2"]),
        Gram1D(["face_###excited###/2"]),
        Gram1D(["left_hello/2"]),
        Gram1D(["left_friend/1"]),
        Gram1D(["left_are/1"]),
        Gram1D(["right_oh/4"]),
        Gram1D(["right_there/1"]),
        Gram1D(["right_friend/1"]),
        Gram1D(["right_you/2"]),
    ],
    't2': [
        Gram1D(["face_happy/2", "face_###excited###/2"]),
        Gram1D(["left_hello/2", "left_friend/1"]),
        Gram1D(["left_friend/1", "left_are/1"]),
        Gram1D(["right_oh/4", "right_there/1"]),
        Gram1D(["right_there/1", "right_friend/1"]),
        Gram1D(["right_friend/1", "right_you/2"]),
    ],
    'c2': [
        Gram1D(["face_happy", "left_hello"]),
        Gram1D(["face_###excited###", "left_hello"]),
        Gram1D(["face_happy", "right_oh"]),
        Gram1D(["face_###excited###", "right_oh"]),
        Gram1D(["left_hello", "right_oh"]),
        Gram1D(["left_friend", "right_friend"]),
        Gram1D(["left_are", "right_you"]),
    ],
    't3': [
        Gram1D(["left_hello/2", "left_friend/1", "left_are/1"]),
        Gram1D(["right_oh/4", "right_there/1", "right_friend/1"]),
        Gram1D(["right_there/1", "right_friend/1", "right_you/2"]),
    ],
    't2c2': [
        Gram2D([["face_happy/2"], ["left_None/1", "left_hello/1"]]),
        Gram2D([["face_happy/1", "face_###excited###/1"], ["left_hello/2"]]),
        Gram2D([["face_###excited###/2"], ["left_hello/1", "left_None/1"]]),
        Gram2D([["face_happy/1", "face_###excited###/1"], ["right_oh/1"]]),
        Gram2D([["face_###excited###/2"], ["right_oh/2"]]),
        Gram2D([["face_###excited###/1", "face_None/1"], ["right_oh/1", "right_there/1"]]),
        Gram2D([["left_None/1", "left_hello/1"], ["right_oh/1"]]),
        Gram2D([["left_hello/2"], ["right_oh/1"]]),
        Gram2D([["left_hello/1", "left_None/1"], ["right_oh/2"]]),
        Gram2D([["left_None/1", "left_friend/1"], ["right_there/1", "right_friend/1"]]),
        Gram2D([["left_friend/1", "left_None/1"], ["right_friend/1", "right_you/1"]]),
        Gram2D([["left_None/1", "left_are/1"], ["right_you/2"]]),
    ]
}


default_2d_grams_nowhitespace_original = {
    't1': [
        Gram1D(["face_happy/2"]),
        Gram1D(["face_###excited###/2"]),
        Gram1D(["left_hello/2"]),
        Gram1D(["left_friend/1"]),
        Gram1D(["left_are/1"]),
        Gram1D(["right_oh/4"]),
        Gram1D(["right_there/1"]),
        Gram1D(["right_friend/1"]),
        Gram1D(["right_you/2"]),
    ],
    't2': [
        Gram1D(["face_happy/2", "face_###excited###/2"]),
        Gram1D(["left_hello/2", "left_friend/1"]),
        Gram1D(["left_friend/1", "left_are/1"]),
        Gram1D(["right_oh/4", "right_there/1"]),
        Gram1D(["right_there/1", "right_friend/1"]),
        Gram1D(["right_friend/1", "right_you/2"]),
    ],
    'c2': [
        Gram1D(["face_happy", "right_oh"]),
        Gram1D(["face_happy", "left_hello"]),
        Gram1D(["face_happy", "right_oh"]),
        Gram1D(["left_hello", "right_oh"]),
        Gram1D(["face_###excited###", "left_hello"]),
        Gram1D(["face_###excited###", "right_oh"]),
        Gram1D(["left_hello", "right_oh"]),
        Gram1D(["face_###excited###", "right_oh"]),
        Gram1D(["left_friend", "right_friend"]),
        Gram1D(["left_are", "right_you"]),
    ],
    't3': [
        Gram1D(["left_hello/2", "left_friend/1", "left_are/1"]),
        Gram1D(["right_oh/4", "right_there/1", "right_friend/1"]),
        Gram1D(["right_there/1", "right_friend/1", "right_you/2"]),
    ],
    't2c2': [
        Gram2D([["face_happy/2"], ["left_None/1", "left_hello/1"]]),
        Gram2D([["face_happy/1", "face_###excited###/1"], ["left_hello/2"]]),
        Gram2D([["face_###excited###/2"], ["left_hello/1", "left_None/1"]]),
        Gram2D([["face_happy/1", "face_###excited###/1"], ["right_oh/1"]]),
        Gram2D([["face_###excited###/2"], ["right_oh/2"]]),
        Gram2D([["face_###excited###/1", "face_None/1"], ["right_oh/1", "right_there/1"]]),
        Gram2D([["left_None/1", "left_hello/1"], ["right_oh/1"]]),
        Gram2D([["left_hello/2"], ["right_oh/1"]]),
        Gram2D([["left_hello/1", "left_None/1"], ["right_oh/2"]]),
        Gram2D([["left_None/1", "left_friend/1"], ["right_there/1", "right_friend/1"]]),
        Gram2D([["left_friend/1", "left_None/1"], ["right_friend/1", "right_you/1"]]),
        Gram2D([["left_None/1", "left_are/1"], ["right_you/2"]]),
    ],
}


class TestGrams:
    def reduce_channels(self, data, channels):
        output = dict()
        for k, grams in data.items():
            output[k] = list()
            for gram in grams:
                gram_channels = gram.get_channels()
                if all(channel in channels for channel in gram_channels):
                    output[k].append(gram)
        return output

    def swap_channels(self, data, swap_map):
        output = dict()
        for k, grams in data.items():
            output[k] = list()
            for gram in grams:
                channel_elements = gram.get_split_elements(strip=False)
                if isinstance(gram, Gram2D):
                    gram_data = [
                        [
                            f'{swap_map.get(channel, channel)}{SEP_KEY}{element}'
                            for channel, element in row
                        ]
                        for row in channel_elements
                    ]
                    new_gram = Gram2D(gram_data)
                else:
                    gram_data = [
                        f'{swap_map.get(channel, channel)}{SEP_KEY}{element}'
                        for channel, element in channel_elements
                    ]
                    new_gram = Gram1D(gram_data)
                output[k].append(new_gram)
        return output

    @pytest.mark.parametrize(
        'data,gram_constructor',
        (
            (["left_g1", "right_g2"], Gram1D),
            ([["left_g1", "right_g2"]], Gram2D),
            (["left_g1", "right_g2"], construct_gram),
            ([["left_g1", "right_g2"]], construct_gram),
        )
    )
    def test_gram_construction(self, data, gram_constructor):
        gram = gram_constructor(data)

    @pytest.mark.parametrize(
        'data,gram_class,elements,strip',
        (
            (["left_g1", "right_g2/2"], Gram1D, ("left_g1", "right_g2/2"), False),
            ([["left_g1", "right_g2/2"]], Gram2D, (("left_g1", "right_g2/2"),), False),
            (["left_g1", "right_g2/2"], construct_gram, ("left_g1", "right_g2/2"), False),
            ([["left_g1", "right_g2/2"]], construct_gram, (("left_g1", "right_g2/2"),), False),
            (["left_g1", "right_g2/2"], Gram1D, ("left_g1", "right_g2"), True),
            ([["left_g1", "right_g2/2"]], Gram2D, (("left_g1", "right_g2"),), True),
            (["left_g1", "right_g2/2"], construct_gram, ("left_g1", "right_g2"), True),
            ([["left_g1", "right_g2/2"]], construct_gram, (("left_g1", "right_g2"),), True),
        )
    )
    def test_gram_elements(self, data, gram_class, elements, strip):
        gram = gram_class(data)
        with catch():
            assert gram.get_elements(strip=strip) == elements

    @pytest.mark.parametrize(
        'data,block_params,gram_params,target',
        (
            (
                data,
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {'time_n': 3, 'channel_n': 2, 'channels': ['face', 'left', 'right']},
                default_2x1d_grams_original,
            ),
            #(
            #    data,
            #    {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
            #    {'time_n': 3, 'channel_n': 2, 'method': '2d', 'whitespace': True, 'channels': ['face', 'left', 'right']},
            #    default_2d_grams_whitespace,
            #),
            (
                data,
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {'time_n': 3, 'channel_n': 2, 'method': '2d', 'channels': ['face', 'left', 'right']},
                default_2d_grams_nowhitespace_original,
            ),
        )
    )
    def test_grams_default(self, data, block_params, gram_params, target):
        blocks = dict_to_block(data, **block_params)
        with catch():
            assert blocks == reduced_blocks
        grams = block_to_gram(blocks, **gram_params)
        with catch():
            assert grams == target

    @pytest.mark.parametrize(
        'data,block_params,gram_params,target',
        (
            (
                data,
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {'time_n': 3, 'channel_n': 2, 'channels': ['left', 'right']},
                default_2x1d_grams_original,
            ),
            (
                data,
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {'time_n': 3, 'channel_n': 2, 'channels': ['right']},
                default_2x1d_grams_original,
            ),
            (
                data,
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {'time_n': 3, 'channel_n': 2, 'channels': ['face']},
                default_2x1d_grams_original,
            ),
            (
                data,
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {'time_n': 3, 'channel_n': 2, 'channels': None},
                default_2x1d_grams_original,
            ),
        )
    )
    def test_grams_channels(self, data, block_params, gram_params, target):
        blocks = dict_to_block(data, **block_params)
        with catch():
            assert blocks == reduced_blocks
        grams = block_to_gram(blocks, **gram_params)
        if gram_params['channels'] is not None:
            target = self.reduce_channels(target, gram_params['channels'])
        # order may be different if channels is None
        grams = {k: sorted(v) for k, v in grams.items()}
        target = {k: sorted(v) for k, v in target.items()}

        with catch():
            assert grams == target

    @pytest.mark.parametrize(
        'data,block_params,gram_params,target',
        (
            (
                data,
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {
                    'time_n': 3,
                    'channel_n': 2,
                    'channels': ['face', 'left', 'right'],
                    'swap_map': {'left': 'right', 'right': 'left'},
                },
                default_2x1d_grams_original,
            ),
            (
                data,
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {
                    'time_n': 3,
                    'channel_n': 2,
                    'channels': ['face', 'left', 'right'],
                    'swap_map': {'left': 'face', 'face': 'left'},
                },
                default_2x1d_grams_original,
            ),
            (
                data,
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {
                    'time_n': 3,
                    'channel_n': 2,
                    'channels': ['face', 'left', 'right'],
                    # not realistic, but allowed:
                    'swap_map': {'left': 'face', 'face': 'right', 'right': 'left'},
                },
                default_2x1d_grams_original,
            ),
            (
                data,
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {
                    'time_n': 3,
                    'channel_n': 2,
                    'channels': ['face', 'left', 'right'],
                    'method': '2d',
                    # not realistic, but allowed:
                    'swap_map': {'left': 'face', 'face': 'right', 'right': 'left'},
                },
                default_2d_grams_nowhitespace_original,
            ),
        )
    )
    def test_grams_swap(self, data, block_params, gram_params, target):
        blocks = dict_to_block(data, **block_params)
        with catch():
            assert blocks == reduced_blocks
        grams = block_to_gram(blocks, **gram_params)
        target = self.swap_channels(target, gram_params['swap_map'])
        with catch():
            assert grams == target

    @pytest.mark.parametrize(
        'data,target_blocks,block_params,gram_params,target',
        (
            (
                data,
                reduced_blocks,
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {
                    'time_n': 3,
                    'channel_n': 2,
                    'channels': ['face', 'left', 'right'],
                },
                default_2x1d_grams_original,
            ),
            (
                data,
                reduced_blocks,
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {
                    'time_n': 3,
                    'channel_n': 2,
                    'channels': ['face', 'left', 'right'],
                    'method': '2d',
                },
                default_2d_grams_nowhitespace_original,
            ),
        )
    )
    def test_grams_original(self, data, target_blocks, block_params, gram_params, target):
        blocks = dict_to_block(data, **block_params)
        with catch():
            assert blocks == target_blocks
        grams = block_to_gram(blocks, **gram_params)
        with catch():
            assert grams == target
