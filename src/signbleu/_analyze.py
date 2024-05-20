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


from metric import time_channel_grams, sign_bleu
from utils import (
    has_value,
    remove_continued,
    extract_manual_sub_blocks,
)
from blockify_dgs import (
    blockify_dgs_file,
    CHANNELS as CHANNELS_DGS,
    combine_dgs_blocks,
)
from blockify_bu import (
    blockify_bu_file,
    combine_bu_blocks,
    ID_SEP,
    NMS as NMS_BU,
    NMS_TRAIN as NMS_TRAIN_BU,
)
from rwlock import RWLock
from block_to_linear import cleanup_linear


channels = [
    "right",
    "left",
    "mouth",
    "eyebrow",
    "head",
]
CHANNELS = channels

NMS = {
    'EBf': 'eyebrow',
    'Hno': 'head',
    'Hs': 'head',
    'Mctr': 'mouth',
    'Mmo': 'mouth',
    'Mo1': 'mouth',
    'Ci': 'mouth',
    'Tbt': 'mouth',
}


def is_nms(gloss):  # need to fix to check for any dataset
    return any(
        gloss.replace('~', '').replace('::', '').startswith(nms)
        for nms in NMS
    )


def nia_get_channel(gloss):
    if gloss.replace('~', '').replace('&', '').startswith('1::'):
        return 'right'
    if gloss.replace('~', '').replace('&', '').startswith('2::'):
        return 'left'
    if gloss.startswith('::') and is_nms(gloss):
        gloss = gloss.replace('::', '')
        if gloss.startswith('Mmo'):
            gloss = 'Mmo'
        return NMS[gloss]
    if re.match(r'(~|&)?[a-zA-Z가-힣]+[0-9]+', gloss):
        return 'both'
    # dynamic sign
    if re.match(r'(~|&)?(날짜|나이|시간|시):.*', gloss):
        return 'both'
    # default finger spelling if not specified
    return 'right'


def dgs_get_channel(gloss):
    if gloss.replace('~', '').replace('&', '').startswith('1::'):
        return 'right'
    if gloss.replace('~', '').replace('&', '').startswith('2::'):
        return 'left'
    if gloss.startswith('::'):  # and is_nms(gloss):
        #gloss = gloss[2:]#.replace('::', '')
        #if gloss == 'Mo1':  # [MG]
        #    return 'Mo1'
        #if gloss == 'Mctr':
        #    return 'Mctr'
        #return 'Mmo'
        return 'mouth'
    if re.match(r'(~|&)?fn.+', gloss):
        return 'right'
    return 'both'


def bu_get_channel(gloss):
    if gloss.replace('~', '').replace('&', '').startswith('1::'):
        return 'right'
    if gloss.replace('~', '').replace('&', '').startswith('2::'):
        return 'left'
    #if is_nms(gloss):  # need to make a bu_is_nms function
    if gloss.startswith('::'):
        gloss = gloss.replace('::', '')
        if gloss in NMS_TRAIN_BU:
            return NMS_TRAIN_BU[gloss]
        elif gloss in NMS_BU:
            return NMS_BU[gloss]
        logger.warn('BU gloss "::{gloss}" not recognized. Returning as a "both" gloss')
        return 'both'
    return 'both'


def get_cont(gloss):
    #return gloss.startswith('~') or is_nms(gloss)
    return gloss.startswith('~') or gloss.startswith('::')


def get_parallel(gloss):
    return gloss.startswith('&')


#def update_previous(items, item_i, item, channel, overlap):
#    if items[item_i].get(channel) is not None and item[channel].replace(':', '') == items[item_i][channel].replace(':', ''):
#        items[item_i][channel] = ':' + items[item_i][channel]


def connect_nms_blocks_old(instance):
    nms_channels = [
        "mouth",
        "eyebrow",
        "head",
    ]
    if len(instance) == 0:
        return instance
    for i in range(len(instance) - 1):
        for channel in nms_channels:
            if channel in instance[i] and channel in instance[i+1] and instance[i][channel].replace(':', '') == instance[i+1][channel].replace(':', ''):
                if instance[i][channel][-1] != ':':
                    instance[i][channel] = instance[i][channel] + ':'
                if instance[i+1][channel][0] != ':':
                    instance[i+1][channel] = ':' + instance[i+1][channel]
    return instance


def connect_nms_blocks(instance, nms_channels=None):
    if len(instance) == 0:
        return instance
    for i in range(len(instance) - 1):
        for channel in instance[i]:
            has_value0 = has_value(instance[i], channel)
            has_value1 = has_value(instance[i+1], channel)
            same = remove_continued(instance[i].get(channel, '')) == \
                remove_continued(instance[i+1].get(channel, ''))
            if has_value0 and has_value1 and same:
                #if instance[i][channel][-1] != ':':
                if not instance[i][channel].endswith(':'):
                    instance[i][channel] = instance[i][channel] + ':'
                #if instance[i+1][channel][0] != ':':
                if not instance[i+1][channel].startswith(':'):
                    instance[i+1][channel] = ':' + instance[i+1][channel]
    return instance


def correct_linear_conflicts(
        instance,
        anchor_pattern='^(~|&)?([12]::)?[~_가-힣]+[0-9]{1,2}(#.*)?$',
        supplement_patterns=None,
):
    if supplement_patterns is None:
        supplement_patterns={
            'mouth': '::(Mctr|Mmo[0-9가-힣]*|Mo1|Ci|Tbt)',
            'head': '::(Hs|Hno)',
            'eyebrow': '::(EBf)',
        }
    seq = instance.split(' ')
    output = list()
    exists = {key: False for key in supplement_patterns}
    for s_i, s in enumerate(seq):
        if re.match(anchor_pattern, s) is not None:
            exists = {k: False for k in exists}  # reset enhancements
            output.append(s)
            continue
        for channel, channel_pattern in supplement_patterns.items():
            if re.match(channel_pattern, s) is not None:
                if not exists[channel]:
                    exists[channel] = True
                    output.append(s)
                continue
    return ' '.join(output)


def _linear_to_block(
        instance,
        get_channel,
        connect_nms=False,
        hands_only=False,
):
    r"""
    Convert a linearized gloss representation to a block representation.

    args:
        instance (str):
        connect_nms (bool, optional): If True, NMS are connected when predicted
            in a series. This should only be set to True when we cannot predict
            continued NMS and when we want to interpret repeated predicitons as
            a single continuous NMS utterance.
        hands_only (bool, optional): If True, use only manual annotations. If
            False, use all channels.
            Defaults to False.
    """
    # instance = correct_linear_conflicts(instance)
    seq = [
        (gloss, get_channel(gloss), get_cont(gloss), get_parallel(gloss))
        for gloss in instance.split(' ')
        #if gloss != ''
    ]
    output = list()
    new_item = dict()
    for gloss, channel, overlap, parallel in seq:
        if gloss == '::':
            continue
        if '::' in gloss:
            gloss = gloss.split('::')[1]
        if channel == 'both':  # new block
            output.append(new_item)
            new_item = {'right': gloss, 'left': gloss}
        #elif not hands_only and is_nms(gloss):
        elif not hands_only and channel not in ('left', 'right'):
            new_item[channel] = gloss
        elif channel in ('left', 'right'):
            opposite = 'left' if channel == 'right' else 'right'
            if channel in new_item or (opposite in new_item and not parallel):
                output.append(new_item)
                new_item = dict()
            new_item[channel] = gloss
            if overlap and len(output) > 0 and opposite in output[-1]:
                new_item[opposite] = ':' + output[-1][opposite]
                output[-1][opposite] = output[-1][opposite] + ':'
    if new_item != dict():
        output.append(new_item)
    if len(output) == 0:
        return output
    if output[0] == dict() and len(output) > 1:
        output = output[1:]
    if connect_nms:
        output = connect_nms_blocks(output)
    return output


def linear_to_block(instance, connect_nms=False, hands_only=False, dataset='nia', cleanup=True):
    dataset = parse_dataset(dataset)
    if cleanup:
        instance = cleanup_linear(instance, dataset=dataset)
    if dataset == 'nia':
        try:
            output = _linear_to_block(
                instance=instance,
                get_channel=nia_get_channel,
                connect_nms=connect_nms,
                hands_only=hands_only,
            )
        except Exception as e:
            raise RuntimeError('errors: ' + repr(e) + str(instance))
    elif dataset == 'dgs':
        output = _linear_to_block(
            instance=instance,
            get_channel=dgs_get_channel,
            connect_nms=connect_nms,
            hands_only=hands_only,
        )
    elif dataset == 'bu':
        output = _linear_to_block(
            instance=instance,
            get_channel=bu_get_channel,
            connect_nms=connect_nms,
            hands_only=hands_only,
        )
    else:
        raise NotImplementedError
    return output


def blockify_text2gloss_results(
        instance,
        file_map,
        offset_threshold=None,
        primary_channels=None,
        connect_nms=True,
        hands_only=False,
):
    text = instance['prompt'].replace('prompt: 한국어를 한국수어로 번역해줘\n\n한국어: ', '').replace('\n한국수어:', '')
    pred = linear_to_block(instance['pred'], connect_nms=connect_nms, hands_only=hands_only)
    refs = [nia_to_block(
        ref,
        offset_threshold=offset_threshold,
        primary_channels=primary_channels,
        hands_only=hands_only,
    ) for ref in file_map[text]]
    output = {
        'prompt': text,
        'pred': pred,
        'ref': refs,
    }
    #output = {
    #    'prompt': instance['prompt'],
    #    'pred': linear_to_block(instance['pred']),
    #    'ref': [linear_to_block(ref) for ref in instance['ref']],
    #}
    return output


#def process_block(block_items, start, end, threshold):
#    r"""
#    Process blocks to remove thresholds (if they exist)
#    """
#    if threshold is None or end - start > threshold:
#        return True
#    lengths = [item['end'] - item['start'] for item in block_items]
#    return any(length <= threshold for length in lengths)


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


def trim_secondary_channels(
        blocks,
        primary_channels,
):
    remove_ids = [
        item_i
        for item_i, item in enumerate(blocks)
        if all(item.get(channel) is None for channel in primary_channels)
    ]
    output = list()
    for idx, block in enumerate(blocks):
        if idx not in remove_ids:
            output.append(block)
        else:
            if idx > 0:
                remove_colons(blocks[idx - 1], side='right')
            if idx < len(blocks) - 1:
                remove_colons(blocks[idx + 1], side='left')
    return output


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


def nia_process_gloss_id(gloss):
    gloss = gloss.strip()
    if gloss == '~':
        gloss = '물결표시0'
    if gloss == '&':
        gloss = 'ampersand0'
    return gloss


def nia_to_block(
        instance,
        offset_threshold=None,
        primary_channels=None,
        hands_only=False,
):
    r"""
    Args:
        instance (dict): NIA data instance to parse.
        offset_threshold (float, optional): If given, gloss start/end times
            will be aligned if their difference is within this threshold.
            Ignored if None.
            Defaults to None.
        primary_channels (tuple[str], optional): A tuple of channels. If not
            None, grams missing all of the specified channels will be ignored.
            Not implemented.
        hands_only (bool, optional): If True, use only manual annotations. If
            False, use all channels.
            Defaults to False.
    """
    start_key = 'start_exact'
    end_key = 'end_exact'
    if not hands_only:
        nms = [
            block
            for key, array in instance['nms_script'].items()
            for block in [{
                'key': key,
                'start': item[start_key],
                'end': item[end_key],
                #'gloss': None if key != 'Mmo' else item['descriptor'],
                'gloss': key,
                'express': None,
            } for item in array]
        ]
    else:
        nms = list()
    ms = [
        block
        for key, array in instance['sign_script'].items()
        for block in [{
            'key': key,
            'start': item[start_key],
            'end': item[end_key],
            'gloss': nia_process_gloss_id(item['gloss_id']['korean']),
            'express': item['express'],
        } for item in array]
    ]
    key_order_map = {
        'sign_gestures_strong': 0,
        'sign_gestures_weak': 1,
    }
    items_pre = sorted(ms + nms, key=lambda x: (x['start'], key_order_map.get(x['key'], 100)))
    items = list()
    times = list()
    for item in items_pre:
        if item['key'] == 'sign_gestures_both':
            new_item = deepcopy(item)
            new_item['key'] = 'sign_gestures_strong'
            items.append(new_item)
            new_item = deepcopy(item)
            new_item['key'] = 'sign_gestures_weak'
            items.append(new_item)
        else:
            items.append(item)
        times.extend((item['start'], item['end']))
    times = sorted(list(set(times)))
    output = list()
    key_map = {
        'sign_gestures_strong': 'right',
        'sign_gestures_weak': 'left',
        'Ci': 'mouth',
        'Hs': 'head',
        'Hno': 'head',
        'EBf': 'eyebrow',
        'Mmo': 'mouth',
        'Mo1': 'mouth',
        'Tbt': 'mouth',
        'Mctr': 'mouth',
    }

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
            channel = key_map[item['key']]
            gloss = item['gloss']
            if item['start'] < times[time_i]:
                gloss = ':' + gloss
            if item['end'] > times[time_i + 1]:
                gloss = gloss + ':'
            block[channel] = gloss
        output.append(block)
    if primary_channels is not None:  # this can be improved
        output = trim_secondary_channels(output, primary_channels)
    return output


def get_blocks(file_map_path, data_path):
    with open(file_map_path, 'rb') as f:
        file_map = json.load(f)
    with open(data_path, 'rb') as f:
        data = json.load(f)
    return [blockify_text2gloss_results(instance, file_map=file_map) for instance in data]


def visualize(
        blocks = None,
        path = 'results.json',
        map_path = 'nia_map.json',
):
    if blocks is None:
        blocks = get_blocks(file_map_path=map_path, data_path=path)


def demo(
        blocks = None,
        time_n = 4,
        channel_n = 4,
        whitespace = True,
        #method = '2d',
        #max_gram_key = 'x4y4',
        method = '2x1d',
        max_gram_key = '4',
        path = 'results.json',
        map_path = 'nia_map.json',
        # path = 'resign_data.json',
        # map_path = 'nia_map_resign.json',
        ignore_none_grams = True,
):
    if blocks is None:
        blocks = get_blocks(file_map_path=map_path, data_path=path)

    pred_grams = [
        time_channel_grams(
            block['pred'],
            time_n=time_n,
            channel_n=channel_n,
            channels=channels,
            method=method,
            whitespace=whitespace,
        )
        for block in tqdm(blocks)
    ]
    ref_grams = [
        [
            (time_channel_grams(
                block['ref'][i],
                time_n=time_n,
                channel_n=channel_n,
                channels=channels,
                method=method,
                whitespace=whitespace,
            )) if len(block['ref']) > i else None
            for block in tqdm(blocks)
        ]
        for i in range(3)
    ]
    pred_lengths = None
    ref_lengths = None
    results = sign_bleu(
        pred_grams,
        ref_grams,
        pred_lengths=pred_lengths,
        ref_lengths=ref_lengths,
        max_gram_key=max_gram_key,
        ignore_none_grams=ignore_none_grams,
    )
    pprint(results)
    return results


class NS21Mapper:
    def __init__(
            self,
            id_file='data_map_id.json',
            text_file='data_map_text.json',
    ):
        with open(id_file, 'rb') as f:
            self.id_map = json.load(f)
        with open(text_file, 'rb') as f:
            self.text_map = json.load(f)

    def from_id(self, idx):
        return self.id_map.get(idx)#, [None])[0]

    def from_text(self, text):
        return self.text_map.get(text)#, [None])[0]


class NS21MapperDir:
    def __init__(
            self,
            id_dir='data_map/nia_data_map_id/',
            text_dir='data_map/nia_data_map_text/',
    ):
        self.id_dir = id_dir
        self.text_dir = text_dir

    def from_id(self, idx):
        with open(os.path.join(self.id_dir, idx+'.json'), 'rb') as f:
            return json.load(f)

    def from_text(self, text):
        key = sha256()
        key.update(text.encode())
        key = key.hexdigest()
        with open(os.path.join(self.text_dir, key+'.json'), 'rb') as f:
            return json.load(f)


class ELANMapper:
    def __init__(self, data_dir, split_path, cache_size, id_sep=None, for_train=False):
        self.data_dir = data_dir
        self.cache = dict()
        self.cache_ref = dict()
        self.lock = RWLock()
        self.cache_size = cache_size
        with open(split_path, 'rb') as f:
            self.split_table = json.load(f)
        if 'seed' in self.split_table:
            del self.split_table['seed']
        self.annot_to_doc = dict()
        for ids in self.split_table.values():
            for doc_id, annot_ids in ids.items():
                for annot_id in annot_ids:
                    self.annot_to_doc[annot_id] = doc_id
        self.id_sep = id_sep
        self.for_train = for_train

    def _add_to_cache(self, idx, data):
        with self.lock.w_locked():
            now = time.time()
            self.cache_ref[idx] = now
            if idx in self.cache:  # assume data is static
                self.cache[idx]['time'] = now
                return None
            else:
                self.cache[idx] = {'time': now, 'data': data}
            if len(self.cache_ref) > self.cache_size:
                oldest = sorted(list(self.cache_ref.items()), key=lambda x: x[1])
                del_idx = oldest[0][0]
                del self.cache[del_idx], self.cache_ref[del_idx]

    def _get_from_cache(self, idx):
        with self.lock.r_locked():
            return self.cache.get(idx, dict()).get('data')

    def from_id(self, idx):
        if idx[0] == 'a':
            annot_id = idx
            doc_id = self.annot_to_doc[annot_id]
        else:
            annot_id = None
            doc_id = idx
        data = self._get_from_cache(doc_id)
        if data is None:
            file_name = doc_id+'.eaf'
            data = self.blockify_file_fn(
                os.path.join(self.data_dir, file_name),
                process_glosses=True,
                return_as='table',
                for_train=self.for_train,
            )
            self._add_to_cache(doc_id, data)
        if annot_id is not None:
            return data[annot_id]
        return self.combine_blocks_fn(data.values())

    def from_text(self, text):
        raise NotImplementedError


class DGSMapper(ELANMapper):
    def __init__(
            self,
            data_dir='data/dgs_samples/',
            split_path='data/dgs_split.json',
            cache_size=1000,
            blockify_file_fn=blockify_dgs_file,
            combine_blocks_fn=combine_dgs_blocks,
            for_train=False,
    ):
        super().__init__(
            data_dir=data_dir,
            split_path=split_path,
            cache_size=cache_size,
            for_train=for_train,
        )
        self.blockify_file_fn = blockify_file_fn
        self.combine_blocks_fn = combine_blocks_fn


class BUMapper(ELANMapper):
    def __init__(
            self,
            data_dir='data/bu_samples/',
            split_path='data/bu_split.json',
            cache_size=1000,
            blockify_file_fn=blockify_bu_file,
            combine_blocks_fn=combine_bu_blocks,
            id_sep=ID_SEP,
            for_train=False,
    ):
        assert id_sep is not None
        super().__init__(
            data_dir=data_dir,
            split_path=split_path,
            cache_size=cache_size,
            id_sep=id_sep,
            for_train=for_train,
        )
        self.blockify_file_fn = blockify_file_fn
        self.combine_blocks_fn = combine_blocks_fn

    def from_id(self, idx):
        # BU data does not use unique annotation IDs, so they are bundled with doc IDs.
        doc_id, annot_id = idx.split(self.id_sep)
        data = self._get_from_cache(doc_id)
        if data is None:
            file_name = doc_id+'.eaf'
            data = self.blockify_file_fn(
                os.path.join(self.data_dir, file_name),
                process_glosses=True,
                return_as='table',
                for_train=self.for_train,
            )
            self._add_to_cache(doc_id, data)
        return data[idx]


# DATA_MAPPER = NS21Mapper()
DATA_MAPPER = NS21MapperDir()
DGS_DATA_MAPPER = DGSMapper()
BU_DATA_MAPPER = BUMapper()
### making a separate copy of mappers to match the preprocessing done by the training scripts to
### allow for reproducibility of previous resuls. This should be combined/cleaned up in the future.
### (I especially don't like having to pass 'for_train' flags to every function)
DGS_DATA_MAPPER_TRAIN = DGSMapper(for_train=True)
BU_DATA_MAPPER_TRAIN = BUMapper(for_train=True)


def parse_dataset(dataset):
    if dataset is None:
        dataset = 'nia'
    known_datasets = ['nia', 'dgs', 'bu']
    message = (
        f'Unknown dataset specifier should be None (defaults to "nia") or in '
        f'{known_datasets}.'
    )
    assert dataset in known_datasets, message
    return dataset


def nia_id_to_block(idx, offset_threshold=None, primary_channels=None, hands_only=False):
    nia_data = DATA_MAPPER.from_id(idx)[0]
    return nia_to_block(
        nia_data,
        offset_threshold=offset_threshold,
        primary_channels=primary_channels,
        hands_only=hands_only,
    )


def dgs_id_to_block(
        idx,
        offset_threshold=None,
        primary_channels=None,
        hands_only=False,
        for_train=False,
):
    if offset_threshold is not None:
        raise NotImplementedError
    if primary_channels is not None:
        raise NotImplementedError
    #if hands_only:
    #    raise NotImplementedError
    if for_train:
        data = DGS_DATA_MAPPER_TRAIN.from_id(idx)['block']
    else:
        data = DGS_DATA_MAPPER.from_id(idx)['block']
    if hands_only:
        data = extract_manual_sub_blocks(data)
    # add post processing here or additional flags to blockify_dgs_file
    return data


def bu_id_to_block(
        idx,
        offset_threshold=None,
        primary_channels=None,
        hands_only=False,
        for_train=False,
):
    if offset_threshold is not None:
        raise NotImplementedError
    if primary_channels is not None:
        raise NotImplementedError
    #if hands_only:
    #    raise NotImplementedError
    if for_train:
        data = BU_DATA_MAPPER_TRAIN.from_id(idx)['block']
    else:
        data = BU_DATA_MAPPER.from_id(idx)['block']
    if hands_only:
        data = extract_manual_sub_blocks(data)
    # add post processing here or additional flags to blockify_dgs_file
    return data


def id_to_block(
        idx,
        offset_threshold=None,
        primary_channels=None,
        hands_only=False,
        dataset='nia',
        for_train=False,
):
    dataset = parse_dataset(dataset)
    if dataset == 'nia':
        return nia_id_to_block(
            idx=idx,
            offset_threshold=offset_threshold,
            primary_channels=primary_channels,
            hands_only=hands_only,
        )
    elif dataset == 'dgs':
        return dgs_id_to_block(
            idx=idx,
            offset_threshold=offset_threshold,
            primary_channels=primary_channels,
            hands_only=hands_only,
            for_train=for_train,
        )
    elif dataset == 'bu':
        return bu_id_to_block(
            idx=idx,
            offset_threshold=offset_threshold,
            primary_channels=primary_channels,
            hands_only=hands_only,
            for_train=for_train,
        )


def nia_id_to_text(idx):
    return DATA_MAPPER.from_id(idx)[0]['text']['raw']['korean']

def id_to_text(idx, dataset='nia'):
    dataset = parse_dataset(dataset)
    if dataset == 'nia':
        return nia_id_to_text(idx)
    elif dataset == 'dgs':
        return DGS_DATA_MAPPER.from_id(idx)['dgs']['text']
    elif dataset == 'bu':
        return BU_DATA_MAPPER.from_id(idx)['bu']['text']


def nia_text_to_block(text, offset_threshold=None, primary_channels=None, hands_only=False):
    nia_data = DATA_MAPPER.from_text(text)[0]
    return nia_to_block(
        nia_data,
        offset_threshold=offset_threshold,
        primary_channels=primary_channels,
        hands_only=hands_only,
    )


def text_to_block(
        text,
        offset_threshold=None,
        primary_channels=None,
        hands_only=False,
        dataset='nia',
):
    dataset = parse_dataset(dataset)
    if dataset == 'nia':
        return nia_text_to_block(
            text=text,
            offset_threshold=offset_threshold,
            primary_channels=primary_channels,
            hands_only=hands_only,
        )
    elif dataset == 'dgs':
        raise NotImplementedError
    elif dataset == 'bu':
        raise NotImplementedError


def instance_count_grams(datum):
    counts = dict()
    for gram in datum:
        gram_ = tuple(gram.get_elements(strip=True))
        if gram_ not in counts:
            counts[gram_] = 0
        counts[gram_] += 1
    return counts


def count_grams(data):
    counts = dict()
    for datum in data:
        counts_ = instance_count_grams(datum)
        for k, c in counts_.items():
            if k not in counts:
                counts[k] = 0
            counts[k] += c
    return counts


def similarity_to_blocks(
        data,
        offset_threshold=None,
        primary_channels=None,
        hands_only=False,
):
    #with open('data_map_id.json', 'rb') as f:
    #    all_map = json.load(f)
    # all_map = DATA_MAPPER.id_map
    output = list()
    for datum in data.values():
        query_file = datum['query']
        other_files = [v for k, v in datum.items() if k != 'query']
        for file in other_files:
            out = dict()
            out['pred'] = nia_to_block(
                DATA_MAPPER.from_id(query_file)[0],
                offset_threshold=offset_threshold,
                primary_channels=primary_channels,
                hands_only=hands_only,
            )
            out['ref'] = [nia_to_block(
                DATA_MAPPER.from_id(file)[0],
                offset_threshold=offset_threshold,
                primary_channels=primary_channels,
                hands_only=hands_only,
            )]
            output.append(out)
    return output


def create_image(blocks, figsize=(25, 5), dpi=400):
    from matplotlib import pyplot as plt
    #from matplotlib import font_manager as fm
    #font_path = r'C:/Users/Windows/Fonts/'
    #fm.findSystemFonts(fontpaths=font_path, fontext='ttf')
    plt.rc('font', family='Malgun Gothic')

    channel_heights = {
        "right": 10,
        "left": 8,
        "mouth": 6,
        "eyebrow": 4,
        "head": 2,
    }
    diff = 2
    eps = 0.1
    fig, ax = plt.subplots(
        **{'figsize': figsize, 'dpi': dpi}
    )
    bar_data = {
        'x': list(),
        'height': list(),
        'width': list(),
        'bottom': list(),
        'fill': False,
        'label': list(),
    }
    channel_length = {
        'right': 0,
        'left': 0,
        'mouth': 0,
        'eyebrow': 0,
        'head': 0,
    }
    for block_i, block in enumerate(blocks):
        for channel, gloss in [(k, v) for k, v in block.items() if v is not None]:
            #p = ax.bar(block_i, width=0.78, height=diff - eps/2, bottom=h-diff + eps/2, fill=False, label=gloss)
            #ax.bar_label(p.get_label(), label_type='center')
            if gloss[-1] == ':':
                channel_length[channel] += 1
                continue
            h = channel_heights[channel]
            bar_data['x'].append(block_i - channel_length[channel]/2)
            bar_data['width'].append(0.78 + channel_length[channel])
            bar_data['height'].append(diff - eps/2)
            bar_data['bottom'].append(h - diff + eps/2)
            bar_data['label'].append(gloss)
            channel_length[channel] = 0
    _ = ax.bar(**bar_data)
    _ = plt.yticks([h-diff/2 for h in channel_heights.values()], channel_heights.keys())
    for x, y, text in zip(bar_data['x'], bar_data['bottom'], bar_data['label']):
        ax.text(
            x,
            y + diff/2,
            s=text,
            fontsize='small',
            horizontalalignment='center',
        )
    # ax.bar_label(p)#, label_type='center')
    plt.savefig('chart_test.png')


if __name__ == '__main__':
    time_n = 4
    channel_n = 4
    whitespace = True
    ignore_none_grams = True
    method = '2d'
    max_gram_key = 'x4y4'
    #method = '2x1d'
    #max_gram_key = '4'
    # path = 'results.json'
    # map_path = 'nia_map.json'
    path = 'resign_data.json'
    map_path = 'nia_map_resign.json'

    if len(sys.argv) > 1 and sys.argv[1] == 'visualize':
        visualize(blocks)
    elif len(sys.argv) > 1 and sys.argv[1] == 'similarity_test':
        with open('similarity_test.json') as f:
            similarity_ref = json.load(f)
        blocks = similarity_to_blocks(similarity_ref)
        results = demo(
            blocks=blocks,
            time_n=time_n,
            channel_n=channel_n,
            whitespace=whitespace,
            method=method,
            max_gram_key=max_gram_key,
            #path=path,
            #map_path=map_path,
            ignore_none_grams=ignore_none_grams,
        )
        output = dict()
        count = 0
        for q_i, question in similarity_ref.items():
            output[q_i] = {
                letter: results['instance_raw'][count + i]
                for i, letter in enumerate([key for key in question.keys() if key != 'query'])
            }
            count += len(question) - 1
        with open(f'similarity_test_results_{method}.json', 'w') as f:
            json.dump(output, f, indent=2)
    else:
        demo(
            time_n=time_n,
            channel_n=channel_n,
            whitespace=whitespace,
            method=method,
            max_gram_key=max_gram_key,
            path=path,
            map_path=map_path,
            ignore_none_grams=ignore_none_grams,
        )
