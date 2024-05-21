# Copyright 2024 EQ4ALL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


r"""
Module for calculating grams.

SignBLEU uses grams of class :class:`Gram` generated with the function
:func:`block_to_gram`\.

"""


import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import re
import json
import copy
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Sequence,
)


import signbleu.constants as CON
from signbleu.utils import (
    powerset,
    add_end_docstrings,
    ARGS_STR,
    NOTE_STR,
)


def set_attr(self, value, attr, default):
    r"""
    Helper function for setting attributes.
    """
    if value is not None:
        self.__setattr__(attr, value)
    else:
        self.__setattr__(attr, default)


class Gram(ABC):
    r"""
    Gram abstract base class.
    """
    def __init__(self, data: Sequence, sep_key: Optional[str] = None):
        r"""
        Initialize :class:`Gram`\.
        """
        set_attr(self, sep_key, "sep_key", default=CON.SEP_KEY)
        self.data = json.dumps(self.dict_sort(data))
        self.reduced = self.get_elements(strip=True)

    def __hash__(self):
        return self.data.__hash__()

    @abstractmethod
    def dict_sort(self, data):
        pass

    @abstractmethod
    def get_elements(self, strip=False, flat=False):
        r"""
        Return the elements in this gram.

        Args:
            strip (bool, optional): If True, the '/n' portion of the elements
                is removed before returning.
            flat (bool, optional): ...
        """
        pass

    @abstractmethod
    def get_channels(self, unique=True):
        r"""
        Return a list of channels covered by this gram.
        """
        pass

    @abstractmethod
    def get_split_elements(self, strip=False):
        r"""
        Return a list of (channel, mod_element) tuples where mod_element is the
        element without the channel identifier.
        """
        pass

    @abstractmethod
    def is_none(self):
        pass

    def count(self, ignore_none=False):
        r"""
        Return the number of elements in the gram.
        """
        elements = self.get_elements(flat=True)
        elements = [e for e in elements if 'None' not in str(e)]
        return len(elements)

    def __str__(self):
        return f"{self.__class__.__name__}({self.reduced})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data})"

    def _class_check(self, other):
        return self.__class__ == other.__class__

    def __eq__(self, other):
        if not self._class_check(other):
            return False
        # return self.data == other.data
        return self.get_elements(strip=True) == other.get_elements(strip=True)

    def __gt__(self, other):
        r"""
        Meaningless greater than comparison. Useful for generating a
        consistent order.
        """
        if not self._class_check(other):
            return False
        return self.data > other.data

    def __ge__(self, other):
        r"""
        Meaningless greater than or equal to comparison. Useful for generating
        a consistent order.
        """
        if not self._class_check(other):
            return False
        return self.data >= other.data

    def __lt__(self, other):
        r"""
        Meaningless less than comparison. Useful for generating a consistent
        order.
        """
        if not self._class_check(other):
            return False
        return self.data < other.data

    def __lt__(self, other):
        r"""
        Meaningless less than or equal to comparison. Useful for generating a
        consistent order.
        """
        if not self._class_check(other):
            return False
        return self.data <= other.data


class Gram1D(Gram):
    r"""
    One-dimensional gram.
    """
    def __init__(self, data, **kwargs):
        r"""
        Initialize :class:`Gram1D`\.
        """
        super().__init__(data=data, **kwargs)
        self.dim = 1

    def dict_sort(self, data):
        try:
            # is row data (temporal gloss) -- do not sort
            [re.search(r'/\d+$', text).span()[0] for text in data]
            return data
        except:
            # is column data (channel gloss)
            return sorted(data)

    def _get_elements(self, data):
        if isinstance(data, str):
            return [data]
        try:
            out = list()
            for row in data:
                out.extend(self._get_elements(row))
            return out
        except:
            return [data]

    def _strip(self, text):
        r"""
        Remove the '/n' suffix from blocked glosses.
        """
        try:
            idx = re.search(r'/\d+$', text).span()[0]
            return text[:idx]
        except:
            return text

    def get_elements(self, strip=False, flat=False):
        r"""
        Return the elements in this gram.

        Args:
            strip (bool, optional): If True, the '/n' portion of the elements
                is removed before returning.
            flat (bool, optional): Not used. Included for compatiblity with
                other gram variants.
        """
        if not hasattr(self, 'elements'):
            self.elements = self._get_elements(json.loads(self.data))
        elements = tuple(self.elements)
        if strip:
            elements = tuple([self._strip(element) for element in elements])
        return elements

    def get_channels(self, unique=True):
        r"""
        Return a list of channels covered by this gram.
        """
        elements = self.get_elements(strip=True)
        channels = [e.split(self.sep_key)[0] for e in elements]
        if unique:
            channels = list(set(channels))
        return channels

    def get_split_elements(self, strip=False):
        r"""
        Return a list of (channel, mod_element) tuples where mod_element is the
        element without the channel identifier.
        """
        elements = self.get_elements(strip=False)
        channels = [
            (e.split(self.sep_key)[0], self.sep_key.join(e.split(self.sep_key)[1:]))
            for e in elements
        ]
        return channels

    def is_none(self):
        return all('None' in e for e in self.get_elements())


class Gram2D(Gram1D):
    r"""
    Two-dimensional gram class.
    """
    def __init__(self, data, **kwargs):
        r"""
        Initialize :class:`Gram2D`\.
        """
        super().__init__(data=data, **kwargs)
        self.dim = 2

    def dict_sort(self, data):
        return sorted(data, key=lambda x: x[0].split(self.sep_key)[0])

    def _get_elements(self, data):
        if isinstance(data, str):
            return ((data,),)
        if isinstance(data, list) or isinstance(data, tuple):
            try:
                if isinstance(data[0], str):
                    return (tuple(data),)
                elif isinstance(data[0], list) or isinstance(data[0], tuple):
                    return tuple([tuple(datum) for datum in data])
                else:
                    raise RuntimeError
            except:
                return (tuple(data),)
        raise RuntimeError

    def get_elements(self, strip=False, flat=False):
        r""" """
        if not hasattr(self, 'elements'):
            self.elements = self._get_elements(json.loads(self.data))
        elements = self.elements
        if strip:
            elements = tuple([
                tuple([self._strip(element) for element in row])
                for row in elements
            ])
        if flat:
            elements = [e for es in elements for e in es]
        return elements

    def is_none(self):
        return all(
            all('None' in e for e in row)
            for row in self.get_elements()
        )

    def get_channels(self, unique=True):
        r"""
        Return a list of channels covered by this gram.
        """
        elements = self.get_elements(strip=True)
        channels = [[e.split(self.sep_key)[0] for e in row] for row in elements]
        if unique:
            channels = list(set([e for row in channels for e in row]))
        return channels

    def get_split_elements(self, strip=False):
        r"""
        Return a list of (channel, mod_element) tuples where mod_element is the
        element without the channel identifier.
        """
        elements = self.get_elements(strip=False)
        channels = [
            [(e.split(self.sep_key)[0], self.sep_key.join(e.split(self.sep_key)[1:])) for e in row]
            for row in elements
        ]
        return channels


# Not convinced that a utility class is the cleanest approach but will save
# some argument passing. Re-evaluate this approach later.
class GramUtilities:
    r"""Utility class for creating and working with grams"""
    def __init__(self, sep_key: Optional[str] = None):
        r"""
        Construct :class:`GramUtilities`\.

        Args:
            sep_key (str, optional): A special str used internally to separate
                channel and gloss names. Can also be specified by setting the
                `SIGNBLEU_SEP_KEY` environment variable.
        """
        set_attr(self, sep_key, "sep_key", default=CON.SEP_KEY)

    def construct_gram(self, data, **kwargs):
        r"""
        Construct new :class:`Gram` object.

        Args:
            data (Sequence): Array data to use as gram. Gram type is automatically
                parsed from the data.
        """
        try:
            inner = data[0]
            if isinstance(inner, list) or isinstance(inner, tuple):
                return Gram2D(data, **kwargs)
            else:
                return Gram1D(data, **kwargs)
        except:
            return Gram1D(data, **kwargs)

    def block_to_array(self, instance, channels):
        if channels is None:
            channels = list(set([
                channel for block in instance for channel in block
            ]))
        return np.array([
            [
                f'{channel}_{block.get(channel)}'
                for channel in channels
            ]
            for block in instance
        ])

    def is_none(self, value):
        return value.split(self.sep_key)[-1].split('/')[0] == 'None'

    def zero_to_one(self, text):
        if text[-1] == '0':
            return text[:-1] + '1'
        return text

    def swap(self, text, swap_map):
        for key in swap_map:
            if text.startswith(key):
                return swap_map[key] + text[len(key):]
        return text

    def combine_consecutive(self, xs):
        output = list()
        cont = ''
        cont_count = 0
        for x_i, x in enumerate(xs):
            if cont == '' and self.is_none(x):
                if x_i == len(xs) - 1:
                    output.append(x.replace(':', '') + '/' + str(cont_count))
                    break
                cont = x
                cont_count = 1
                continue
            if self.is_none(cont) and self.is_none(x):
                cont_count += 1
                if x_i == len(xs) - 1:
                    output.append(x.replace(':', '') + '/' + str(cont_count))
                    break
                continue
            elif self.is_none(cont) and not self.is_none(x):
                output.append(cont.replace(':', '') + '/' + str(cont_count))
                cont = ''
                cont_count = 0

            if x.split(self.sep_key)[1].startswith(':'):  # x should be continuing previous
                if cont.replace(':', '') != x.replace(':', ''):  # not continuing. error
                    logger.debug('Continued tokens not matching')
                    if cont != '':
                        output.append(cont.replace(':', '') + '/' + str(cont_count))
                    cont = ''
                    cont_count = 0
            if x[-1] == ':':  # next token should continue
                if x_i == len(xs) - 1:
                    output.append(x.replace(':', '') + '/' + str(cont_count))
                    break
                else:
                    cont = x
                    cont_count += 1
            else:
                if x.split('_')[1].startswith(':'):  # end continued sign
                    cont_count += 1
                output.append(x.replace(':', '') + '/' + str(cont_count))
                cont = ''
                cont_count = 0

        # should fix in the above, but quick fix here:
        output = [self.zero_to_one(item) for item in output]
        return output

    def x_grams(self, block, n, whitespace, swap_map=None):
        output = {str(i): list() for i in range(1, n + 1)}
        if len(block) == 0:
            return output
        x, y = block.shape
        for col_i in range(y):  # for each column
            col = block[:, col_i]
            col = self.combine_consecutive(col)
            if not whitespace:  # remove whitespace
                col = [item for item in col if not self.is_none(item)]
            if swap_map is not None:  # swap channels
                col = [self.swap(item, swap_map) for item in col]
            length = len(col)
            for n_ in range(1, n + 1):  # calculate n grams
                if length < n_:
                    continue
                output[str(n_)].extend([col[i: i + n_] for i in range(length - n_ + 1)])
        return output

    def y_grams(self, block, n, whitespace, swap_map=None, hand_channels=None):
        start = 2  # unigrams already collected from x_grams()
        output = {str(i): list() for i in range(start, n + 1)}
        for row in block:
            row = [item.replace(':', '') for item in row]
            if not whitespace:
                row = [item for item in row if not self.is_none(item)]
            if swap_map is not None:
                row = [self.swap(item, swap_map) for item in row]
            grams = powerset(row, n, start=start)
            for gram in grams:
                output[str(len(gram))].append(gram)# + '/1')
        return output

    def combine_and_gram_xy(self, block, length=2):
        def combine_glosses(gram):
            gram = np.array(gram)
            out = list()
            for col in gram.T:
                out.append(self.combine_consecutive(col))
            return out
        output = list()
        gram = list()
        row_i = 0
        while row_i < len(block):
            row = block[row_i]
            gloss_stop = any(
                gloss[-1] != ':'
                for gloss in row
            )
            gloss_stop = any(
                gloss[-1] != ':'
                for gloss in row
            )
            gloss_start = any(
                gloss.split('_')[1][0] != ':'
                for gloss in row
            )
            gloss_cont = any(
                gloss.split('_')[1][0] == ':'
                for gloss in row
            )
            if gloss_stop or (gloss_start and gloss_cont):
                if len(gram) == 0:
                    start_i = row_i + 1
                gram.append(row)
            if len(gram) == length:
                output.append(combine_glosses(gram))
                row_i = start_i
                gram = list()
            else:
                row_i += 1
        return output

    # combine consecutive blocks
    def xy_grams_reduced(self, block, x_n, y_n, whitespace, swap_map=None):
        output = {'t2c2': list()}
        # do only a 2x2 gram
        x_n = y_n = 2
        try:
            x, y = block.shape
        except:
            return output
        if x < x_n or y < y_n:
            return output
        y_ind_sets = list(powerset(range(y), y_n, start=2))
        for y_inds in y_ind_sets:
            _grams = self.combine_and_gram_xy(block[:, y_inds], length=x_n)
            for gram in _grams:
                if swap_map is not None:
                    gram = [
                        [self.swap(g, swap_map) for g in channel_grams]
                        for channel_grams in gram
                    ]
                if any(all(self.is_none(gloss) for gloss in channel) for channel in gram):
                    continue
                output['t2c2'].append(gram)
        return output

    def update_gram_collection(self, grams, new_grams):
        for k, gs in new_grams.items():
            if k not in grams:
                grams[k] = list()
            grams[k].extend(gs)

    def combine_hands(self, grams, hand_channels):
        # just changes 'left' and 'right' to the general 'hand' channel.
        # Later need to look into actually combining them so there aren't repetitions.
        def _map_hand(element):
            for hand in hand_channels:
                if element.startswith(hand):
                    try:
                        _ = re.match(r'^.*/\d+$', element).group()
                        element = 'hand' + element[len(hand):]
                    except:
                        element = 'simulhand' + element[len(hand):]
            return element

        grams = {
            k: [
                [
                    _map_hand(element) \
                            if isinstance(element, str) else \
                            [_map_hand(el) for el in element]
                    for element in gram
                ]
                for gram in arr
            ]
            for k, arr in grams.items()
        }
        return grams


GRAM_ID = """
    < : **Gramming Args ->**
"""

BLOCK_TO_GRAM_ARGS = r"""
    instance (list): Block data to gram.
"""

BLOCK_TO_GRAM_PARAMS = r"""
    time_n (int): The max gram size for intra-channel glosses.
    channel_n (int): The max gram size for inter-channel glosses.
        Note that since the channel exis has no inherent ordering, ngrams
        are really just n-sized subsets of co-occuring glosses.
    channels (list[str], optional): The name of channels to use if
        'instance' is a dictionary and you want to specify specific
        channels to use.
        Defaults to None.
    method (str): the ngram method to use. Possible values are 'time'
        (time-dimension ngrams only), 'channel' (channel-dimension ngrams
        only), '1d' (time- and channel-dimension ngrams calculated
        separately), and '2d' (time- and channel-dimension ngrams
        calculated together). Note that method='2d' implies that
        rectangular ngram sets will be returned.
    swap_map (dict, optional): dict specifying channel name swaps. For
        example, to swap the left and right hand channels, use the
        following: `{'left': 'right', 'right': 'left'}`\.
        Defaults to None.
    hand_channels (tuple, optional): Hand channels. Used only if
        combine_hand_channels is True.
        Defaults to ('left', 'right').
    sep_key (str, optional): If specified, uses this separator to separate
        channel and gloss keys in grams. Should be a string that is not used
        in glosses. If None, defaults to "_". Can also be specified using the
        `SIGNBLEU_SEP_KEY` environment variable.
        Defaults to None.
"""

BLOCK_TO_GRAM_NOTES = (r"""
    Grams containing None (whitespace) values are currently not being
    calculated correctly for `original_y_grams=False` with
    `whitespace=True`\. Additionally, `combine_hand_channels=True`
    appears to be suffering from a critical bug and needs to be fixed.
""", )

@add_end_docstrings(
    ARGS_STR,
    BLOCK_TO_GRAM_ARGS,
    BLOCK_TO_GRAM_PARAMS,
    NOTE_STR + NOTE_STR.join(BLOCK_TO_GRAM_NOTES),
)
def block_to_gram(
    instance: List[Dict],
    time_n: int,
    channel_n: int,
    channels: Optional[Tuple[str]] = None,
    method: str = '1d',
    swap_map: Optional[Dict[str, str]] = None,
    hand_channels: Tuple[str] = ('right', 'left'),
    sep_key: Optional[str] = None,
):
    r"""
    Generate ngrams for a single sample.

    """
    whitespace = False
    combine_hand_channels = False
    single_suji_channel = False
    return_separate = True
    original_y_grams = True

    gram_utils = GramUtilities(sep_key=sep_key)

    if single_suji_channel:
        assert not original_y_grams
    instance = copy.deepcopy(instance)
    grams = dict()
    block = gram_utils.block_to_array(instance, channels)  # time x channel array
    if method in ['time', '1d', '2d']:
        time_grams = gram_utils.x_grams(
            block,
            n=time_n,
            whitespace=whitespace,
            swap_map=swap_map,
        )
        if return_separate:
            time_grams = {'t'+k: v for k, v in time_grams.items()}
        gram_utils.update_gram_collection(grams, time_grams)
    if method in ['channel', '1d', '2d']:
        channel_grams = gram_utils.y_grams(
            block,
            n=channel_n,
            whitespace=whitespace,
            swap_map=swap_map,
        )
        if return_separate:
            channel_grams = {'c'+k: v for k, v in channel_grams.items()}
        gram_utils.update_gram_collection(grams, channel_grams)
    if method == '2d' and time_n > 1 and channel_n > 1:
        xy_gram_results = gram_utils.xy_grams_reduced(
            block,
            x_n=time_n,
            y_n=channel_n,
            whitespace=whitespace,
            swap_map=swap_map,
        )
        gram_utils.update_gram_collection(grams, xy_gram_results)
    if combine_hand_channels or single_suji_channel:
        grams = gram_utils.combine_hands(grams, hand_channels=hand_channels)
    grams = {
        key: list(filter(
            lambda x: not x.is_none(),
            [
                gram_utils.construct_gram(g, sep_key=sep_key)
                for g in gs
            ],
        ))
        for key, gs in grams.items()
    }
    return grams
