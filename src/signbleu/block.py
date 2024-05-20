r"""
Module for blocking data.

:func:`dict_to_block` and :func:`list_to_block` are the main functions for
converting annotation data into blocked data.
"""

import re
import os
import sys
import time
import json
import numpy as np
from tqdm import tqdm
from hashlib import sha256
from pprint import pprint
from copy import deepcopy
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import signbleu.constants as CON
from signbleu.utils import (
    add_end_docstrings,
    ARGS_STR,
    NOTE_STR,
)


def align_blocks(
        items,
        times,
        offset_threshold,
):
    time_i = 0
    while time_i < len(times)-1:
        overlap = [
            item
            for item in items
            if item['start'] <= times[time_i] and item['end'] >= times[time_i + 1]
        ]
        items, times = align_blocks_by_threshold(
            overlap,
            times[time_i],
            times[time_i+1],
            offset_threshold,
            all_items=items,
            all_times=times,
        )
        time_i += 1
    return items, times


def align_blocks_by_threshold(block_items, start, end, threshold, all_items, all_times):
    # can cause collapse in edge case with large threshold
    segment = end - start

    # if no threshold or segment longer than threshold, do nothing
    if threshold is None or segment > threshold:
        return all_items, all_times

    # if any gloss extends for exactly this segment, do nothing
    lengths = [item['end'] - item['start'] for item in block_items]
    if any(length == segment for length in lengths):
        return all_items, all_times

    # else, shift all matching right
    output = list()
    all_times = [time for time in all_times if time != start]
    for item in all_items:
        if item['end'] in [start, end]:
            item['end'] = end
        if item['start'] in [start, end]:
            item['start'] = end
        output.append(item)
    return output, all_times


def remove_colons(d, side):
    keys = list(d.keys())
    for key in keys:
        value = d[key]
        if value is None:
            continue
        if side == 'left' and value.startswith(':'):
            value = value[1:]
        if side == 'right' and value[-1] == ':':
            value = value[:-1]
        d[key] = value


# I don't think that key_order is necessary. Double check.
def list_to_block(
    data: List[Dict],
    offset_threshold: Optional[float] = None,
    key_order: Optional[Dict] = None,
    two_hand_map: Optional[Dict[str, List[str]]] = None,
    channel_combine_map: Optional[Dict[str, str]] = None,
    mask_key: Optional[str] = None,
    copy_data: bool = True,
):
    if mask_key is None:
        mask_key = CON.MASK_KEY
    if copy_data:
        data = deepcopy(data)
    if key_order is None:
        key_order = dict()
    data = sorted(
        data,
        key=lambda x: (x['start'], key_order.get(x['channel'], x['channel'])),
    )
    if two_hand_map is None:
        two_hand_map = dict()
    if channel_combine_map is None:
        channel_combine_map = dict()
    items = list()
    times = list()

    # modify annotations
    for item in data:
        item['channel'] = channel_combine_map.get(
            item['channel'],
            item['channel'],
        )
        if item['gloss'].startswith(':'):
            item['gloss'] = mask_key + item['gloss']
        if item['gloss'].endswith(':'):
            item['gloss'] = item['gloss'] + mask_key

    for item in data:
        if item['channel'] in two_hand_map:
            for hand in two_hand_map[item['channel']]:
                new_item = deepcopy(item)
                new_item['channel'] = hand
                items.append(new_item)
        else:
            items.append(item)
        times.extend((item['start'], item['end']))
    times = sorted(list(set(times)))

    channels = set([item['channel'] for item in items])

    output = list()

    items, times = align_blocks(
        items,
        times,
        offset_threshold,
    )

    for time_i in range(len(times) - 1):
        overlap = [
            item
            for item in items
            if item['start'] <= times[time_i] and item['end'] >= times[time_i + 1]
        ]
        if overlap is None or len(overlap) == 0:
            continue
        block = {channel: None for channel in channels}
        for item in overlap:
            if block[item['channel']] is not None:
                message = (
                    f'Time overlap found for channel "{item["channel"]}" at time'
                    f' interval ({times[time_i]}, {times[time_i+1]}).'
                )
                raise RuntimeError(message)
            gloss = item['gloss']
            if item['start'] < times[time_i]:
                gloss = ':' + gloss
            if item['end'] > times[time_i + 1]:
                gloss = gloss + ':'
            block[item['channel']] = gloss
        output.append(block)
    return output


BLOCK_ID = """
    < : **Blocking Args ->**
"""
DICT_TO_BLOCK_ARGS = """
    data (dict): Data to parse of form {channel: [{}, {}, ...]}
"""
DICT_TO_BLOCK_PARAMS = """
    offset_threshold (float, optional): If given, time segments within
        `offset_threshold` of eachother are joined.
        This may be useful for removing small annotation offset errors.
        Defaults to None.
    channel_keys (tuple[str], optional): If given, only signals from
        the given channels will be used.
        Defaults to None.
    start_key (str, optional): The key mapped to signal start time.
        Defaults to "start".
    end_key (str, optional): The key mapped to signal end time.
        Defaults to "end".
    gloss_key (str, optional): The key mapped to gloss or gesture name.
        Defaults to "gloss".
    two_hand_map (dict[str, list[str]], optional): Specifies 1:2 tier mappings
        for easy assignment of annotations in a double-hand tier to two
        single-hand tiers.
        For example, `two_hand_map`\={"both": ["left", "right"]} will copy all
        annotations in "both" to the left and right channels.
        Ignored if None.
        Defaults to None.
    channel_combine_map (dict[str, str], optional): Specifies n:1 tier
        mappings for combining multiple tiers into a single channel.
        For example, `channel_combine_map`\={"lip_smile": "mouth", "lip_frown":
        "mouth"} will map annotations in both "lip_smile" and "lip_frown" tiers
        to the "mouth" channel.
        Note that combining channels will throw an error if annotations from
        any source tier overlap in the combined channel.
        Ignored if None.
        Defaults to None.
    mask_key (str, optional): The string appended to the start or end of
        tokens that start or end, respectively, with a colon. SignBLEU uses
        colons to denote when glosses continue across blocks. The default
        mask key is '###', but if your dataset uses '###' in their glosses,
        a different mask key can be set with a parameter to this function
        or by setting the `SIGNBLEU_MASK_KEY` environment variable.
"""

@add_end_docstrings(
    BLOCK_ID,
    ARGS_STR,
    DICT_TO_BLOCK_ARGS,
    DICT_TO_BLOCK_PARAMS,
)
def dict_to_block(
    data: Dict[str, List],
    offset_threshold: Optional[float] = None,
    channel_keys: Optional[Tuple[str]] = None,
    start_key: str = 'start',
    end_key: str = 'end',
    gloss_key: str = 'gloss',
    two_hand_map: Optional[Dict[str, List[str]]] = None,
    channel_combine_map: Optional[Dict[str, str]] = None,
    mask_key: Optional[str] = None,
):
    r"""
    A convenience wrapper for list_to_block.

    Convert dict of form
    {channel: [{'gloss': 'g1', 'start': 0.0, 'end': 1.0}, ...]} to blocks.

    """
    if channel_keys is None:
        channel_keys = tuple(data.keys())
    key_order = {
        k: i
        for i, k in enumerate(channel_keys)
    }
    signs = list()
    for key in channel_keys:
        for sign in data[key]:
            signs.append({
                'channel': key,
                'start': sign[start_key],
                'end': sign[end_key],
                'gloss': sign[gloss_key],
            })
    return list_to_block(
        data=signs,
        offset_threshold=offset_threshold,
        key_order=key_order,
        two_hand_map=two_hand_map,
        channel_combine_map=channel_combine_map,
        copy_data=False,
        mask_key=mask_key,
    )
