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
Module for calculating SignBLEU.

SignBLEU can be called with :func:`signbleu`\ or using the more user-friendly
:class:`SignBLEU`\'s :meth:`calculate` method.

See :func:`signbleu.block_to_gram` for calculating grams for SignBLEU.

"""


import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import re
import copy
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union
from collections.abc import Sequence
import sys
if sys.version_info[1] < 9:
    # workaround to support python3.8 -- planned to support through end of 2024
    from typing import Sequence as SequenceHint
else:
    SequenceHint = Sequence


from signbleu.shapley import marginal_count
from signbleu.utils import (
    add_start_docstrings,
    add_end_docstrings,
    add_start_init_docstrings,
    add_end_init_docstrings,
    ARGS_STR,
    NOTE_STR,
)
from signbleu.block import (
    dict_to_block,
    BLOCK_ID,
    DICT_TO_BLOCK_PARAMS,
)
from signbleu.outputs import (
    Output,
    OutputConstructor,
    Signature,
)
from signbleu.gram import (
    GRAM_ID,
    Gram,
    block_to_gram,
    BLOCK_TO_GRAM_PARAMS,
)


def count_glosses_from_blocks(blocks, channels=None):
    count = 0
    for block in blocks:
        if channels is None:
            values = block.values()
        else:
            values = [
                block[channel]
                for channel in channels
                if block.get(channel) is not None
            ]
        for value in values:
            if value is None:
                continue
            if not value.endswith(':'):
                count += 1
    return count


def validate_signbleu_inputs(preds, refs, pred_lengths, ref_lengths):
    assert isinstance(preds, Sequence)
    len_preds = len(preds)
    assert all(isinstance(pred, dict) for pred in preds)

    assert isinstance(refs, Sequence)
    for ref_list in refs:
        assert isinstance(ref_list, Sequence)
        assert all(isinstance(ref, dict) or ref is None for ref in ref_list)
        assert len(ref_list) == len_preds

    assert len(pred_lengths) == len(ref_lengths) == len_preds
    assert all(isinstance(length, int) for length in pred_lengths)
    assert all(isinstance(length, int) for length in ref_lengths)


def safe_div(a, b, default=0):
    if b == 0:
        return default
    return a / b


def get_numerator(
    pred_gram_count: dict,
    ref_gram_count: dict,
):
    return [
        min(pc, ref_gram_count.get(g, 0))
        for g, pc in pred_gram_count.items()
    ]


def get_denominator(pred_gram_count):
    return [pc for g, pc in pred_gram_count.items()]


def no_matching(data):
    data = np.array([scores['num'] for scores in data.values()])
    if data.size == 0:
        return list()
    return np.where(data.sum(axis=0) == 0)[0]

def smooth_sentence_scores(bleu_n, smoothing, epsilon=0.1):
    skip_inds = no_matching(bleu_n)
    bleu_n = copy.deepcopy(bleu_n)
    if smoothing == 'epsilon':
        for n, vals in bleu_n.items():
            for i in range(len(vals['num'])):
                if vals['den'][i] == 0:
                    continue
                if vals['num'][i] == 0:
                    vals['num'][i] += epsilon
    elif smoothing == 'add_k':  # 'add-k' in sacrebleu
        for n, vals in bleu_n.items():
            if n in ['1', 't1']:  # apply only for N >= 2
                continue
            vals['num'] = [v+1 for v in vals['num']]
            vals['den'] = [v+1 for v in vals['den']]
    elif smoothing == 'add_k_all':  # same as 2 but apply to n=1 too
        for n, vals in bleu_n.items():
            vals['num'] = [v+1 for v in vals['num']]
            vals['den'] = [v+1 for v in vals['den']]
    elif smoothing == 'exponential':  # 'exp' in sacrebleu
        invcnt = [1 for _ in range(len(bleu_n.get('t1', {'num': list()})['num']))]
        for n, vals in bleu_n.items():
            for i in range(len(vals['num'])):
                if vals['den'][i] == 0:
                    continue
                if vals['num'][i] == 0:
                    invcnt[i] *= 2
                    vals['num'][i] = 1 / invcnt[i]
    elif smoothing == 'floor':  # 'floor' in sacrebleu
        #invcnt = [1 for _ in range(len(bleu_n.get('1', {'num': list()})['num']))]
        for n, vals in bleu_n.items():
            for i in range(len(vals['num'])):
                if vals['den'][i] == 0:
                    continue
                if vals['num'][i] == 0:
                    #invcnt[i] *= 2
                    #vals['num'][i] = 1 / invcnt[i]
                    vals['num'][i] = epsilon
    elif smoothing is not None and smoothing != 'none':
        raise NotImplementedError
    for scores in bleu_n.values():
        for i in skip_inds:
            scores['num'][i] = 0
    return bleu_n


def standardize_weights(weights):
    weight_sum = sum(weights.values())
    return {k: v/weight_sum for k, v in weights.items()}


SIGNBLEU_ID = r"""
    < : **SignBLEU Args ->**
"""
SIGNBLEU_ARGS = r"""
    hypotheses (list[dict[str, list[:class:`Gram`\]]]): Hypothesis grams as
        [hyp1, hyp2, ...] where hypi is a list of grams.
    references (list[list[dict[str, list[:class:`Gram`\]]]]): Reference grams as
        [[hyp1_ref1, hyp2_ref1, ...], [hyp1_ref2, hyp2_ref2, ...], ...]
        where hypi_refj is a list of grams.
    hyp_lengths (list[int]): Hypothesis lengths.
    ref_lengths (list[int]): Best reference lengths. This should
        be a list of ints representing the length of the reference of most
        similar length to its hypothesis.
"""
SIGNBLEU_PARAMS = r"""
    smoothing (str, optional): The smoothing method to use for sentence
        SignBLEU. See the note below for additional information. Possible
        values are: None, "none", "epsilon", "add_k", "exponential", and
        "floor".
        Defaults to "exponential".
    effective_order (bool, optional): If True, include n-gram counts only for
        precision > 0 when calculating sentence SignBLEU.
        Defaults to True.
    verbose (bool, optional): If True, print weights and progress bars.
        Defaults to False
"""

SIGNBLEU_NOTES = ("""
    We use string values to denote types of smoothing to avoid using what may
    appear as magic numbers. Names are mostly taken from sacrebleu.
    See "A Systematic Comparison of Smoothing Techniques for Sentence-Level
    BLEU" for more details:
    https://statmt.org/wmt14/pdf/W14-3346.pdf:

    * None or "none": No smoothing
    * "epsilon" (method #1) add epsilon to the numerator when num == 0 & den != 0
    * "add_k" (method #2) add 1 to the numerator and denominator when N > 1
    * "exponential" (method #3) numerator from geometric sequence when num == 0 and den != 0
    * "floor" from sacrebleu when num == 0 and den != 0

""", )

@add_end_docstrings(
    ARGS_STR,
    SIGNBLEU_ARGS,
    SIGNBLEU_PARAMS,
    NOTE_STR + NOTE_STR.join(SIGNBLEU_NOTES),
)
def signbleu(
    hypotheses: SequenceHint[Dict[str, SequenceHint[Gram]]],
    references: SequenceHint[SequenceHint[Dict[str, SequenceHint[Gram]]]],
    hyp_lengths: SequenceHint[int],
    ref_lengths: SequenceHint[int],
    smoothing: str = 'exponential',
    weights: Optional[Dict[str, float]] = None,
    effective_order: bool = True,
    verbose = False,
):
    r"""
    Calculates SignBLEU from grams.

    """

    ignore_none_grams = True

    validate_signbleu_inputs(hypotheses, references, hyp_lengths, ref_lengths)

    if weights is None:
        weights = dict()
    else:
        weights = standardize_weights(weights)

    # drop all-None grams
    preds = [
        {key: [g for g in gs if not g.is_none()] for key, gs in pred_n.items()}
        for pred_n in hypotheses
    ]
    refs = [
        [
            {key: [g for g in gs if not g.is_none()] for key, gs in ref_instance.items()} if ref_instance is not None else None
            for ref_instance in ref_set
        ]
        for ref_set in references
    ]

    bleu_n_pre = dict()
    if verbose:
        pbar = tqdm(total=len(preds))
    for i in range(len(preds)):
        instance = preds[i]
        ref_gram_count = dict()
        pred_gram_count = dict()
        for n, inst_grams in instance.items():
            if n not in bleu_n_pre:
                bleu_n_pre[n] = {'num': list(), 'den': list()}
            ref_grams = [
                [gram for gram in ref[i][n] if gram in inst_grams]
                for ref in refs
                if ref[i] is not None
            ]
            ref_grams_counts = [
                {k: gram for k, gram in zip(*np.unique(ref, return_counts=True))}
                for ref in ref_grams
            ]
            ref_gram_count[n] = dict()
            for gram_count_sub in ref_grams_counts:
                for gram, count in gram_count_sub.items():
                    if count > ref_gram_count[n].get(gram, 0):
                        ref_gram_count[n][gram] = count
            pred_gram_count[n] = {g: c for g, c in zip(*np.unique(inst_grams, return_counts=True))}
        for n in instance:
            bleu_n_pre[n]['num'].append(sum(get_numerator(
                pred_gram_count=pred_gram_count[n],
                ref_gram_count=ref_gram_count[n],
            )))
            bleu_n_pre[n]['den'].append(sum(get_denominator(
                pred_gram_count=pred_gram_count[n],
            )))
        if verbose:
            pbar.update()
    bleu_n = dict()
    for n in bleu_n_pre:
        #sum_bleu_n_pre_den = sum(bleu_n_pre[n]['den'])
        #if sum_bleu_n_pre_den == 0:
        #    bleu_n[n] = 0
        #else:
        #    bleu_n[n] = sum(bleu_n_pre[n]['num']) / sum_bleu_n_pre_den
        # bleu_n[n] = sum(bleu_n_pre[n]['num']) / sum(bleu_n_pre[n]['den'])
        #if sum(bleu_n_pre[n]['den']) == 0:
        #    breakpoint()
        bleu_n[n] = safe_div(sum(bleu_n_pre[n]['num']), sum(bleu_n_pre[n]['den']), 0)

    corpus_score = np.sum([
        np.log(score) / weights.get(n, len(bleu_n)) if score != 0 else -float('inf')
        for n, score in bleu_n.items()
    ])
    corpus_score = np.exp(corpus_score)# if corpus_score != 0 else 0

    if smoothing is not None and smoothing != 0:
        bleu_n_smooth = smooth_sentence_scores(bleu_n_pre, smoothing)
    else:
        bleu_n_smooth = bleu_n_pre
    if effective_order:
        instance_score = list()
        for idx in range(len(preds)):
            sum_ = list()
            for n, num_den in bleu_n_smooth.items():
                if n == '1' and num_den['den'][idx] == 0:  # if n=1 is 0, all 0
                    instance_score.append(0)
                    break
                elif num_den['den'][idx] == 0:  # if n>1 is 0, ignore
                    continue
                # how to handle warnings here
                sum_.append(
                    #np.log(num_den['num'][idx] / num_den['den'][idx]) / weights.get(n, len(bleu_n))
                    np.log(num_den['num'][idx] / num_den['den'][idx])
                )
            else:
                #instance_score.append(np.sum(sum_))
                sum_ = np.array(sum_) / len(sum_)  # ignore manual weights
                instance_score.append(sum_.sum())
    else:
        instance_score = [
            np.sum([
                np.log(num_den['num'][idx] / num_den['den'][idx]) / weights.get(n, len(bleu_n)) \
                    if num_den['den'][idx] != 0 else -float('inf')
                for n, num_den in bleu_n_smooth.items()
            ])
            for idx in range(len(preds))
        ]
    instance_score = [
        np.exp(_score)# if _score != 0 else 0
        for _score in instance_score
    ]

    pred_lengths = hyp_lengths
    instance_bp = [
        # np.exp(1 - r / p) if p <= r else 1 for p, r in zip(pred_lengths, ref_lengths)
        np.exp(1 - safe_div(r, p, 1_000)) if p <= r else 1 for p, r in zip(pred_lengths, ref_lengths)
    ]
    r = sum(ref_lengths)
    p = sum(pred_lengths)
    # corpus_bp = np.exp(1 - r / p) if p <= r else 1
    corpus_bp = np.exp(1 - safe_div(r, p, 1_000)) if p <= r else 1

    return {
        'corpus_raw': corpus_score,
        'instance_raw': instance_score,
        'corpus_bp': corpus_bp,
        'instance_bp': instance_bp,
        'corpus_signbleu': corpus_bp * corpus_score,
        'instance_signbleu': [score * bp for score, bp in zip(instance_score, instance_bp)],
    }


def get_similar_length(length, candidate_lengths):
    diffs = [
        (i, abs(length - candidate))
        for i, candidate in enumerate(candidate_lengths)
        if candidate is not None
    ]
    closest = sorted(diffs, key=lambda x: x[1])[0][0]
    return candidate_lengths[closest]


@add_end_init_docstrings(
    # args
    ARGS_STR,
    BLOCK_ID,
    DICT_TO_BLOCK_PARAMS,
    GRAM_ID,
    BLOCK_TO_GRAM_PARAMS,
    SIGNBLEU_ID,
    SIGNBLEU_PARAMS,
    # notes
    NOTE_STR + NOTE_STR.join(SIGNBLEU_NOTES),
)
class SignBLEU:
    r"""
    Convenience class for calculating SignBLEU from neutral or block data.
    """
    def __init__(  # allow setting all params here?
        self,
        # block params
        offset_threshold: Optional[float] = None,
        channel_keys: Optional[Tuple[str]] = None,
        start_key: str = 'start',
        end_key: str = 'end',
        gloss_key: str = 'gloss',
        two_hand_map: Optional[Dict[str, List[str]]] = None,
        channel_combine_map: Optional[Dict[str, str]] = None,
        mask_key: Optional[str] = None,
        # gram params
        time_n: int = 3,
        channel_n: int = 2,
        channels: Optional[Tuple[str]] = None,
        method: str = '1d',
        swap_map: Optional[Dict[str, str]] = None,
        hand_channels: Tuple[str] = ('right', 'left'),
        sep_key: Optional[str] = None,
        # metric params
        smoothing = 'exponential',
        effective_order: bool = True,
        verbose = False,
    ):
        r"""
        Initialize :class:`SignBLEU`\.

        Permanent parameters for block and signbleu calculation can be set
        here, but will be overridden by parameters set in the :meth:`_block`,
        :meth:`_gram`\, and :meth:`_signbleu` methods when called directly.

        Example:
            >>> block_params = {...}
            >>> gram_params = {...}
            >>> signbleu_params = {...}
            >>> METRIC = SignBLEU(**block_params, **gram_params, **signbleu_params)
            >>> score = METRIC.calculate(predictions, references)

        """
        self.block_params = {
            'offset_threshold': offset_threshold,
            'channel_keys': channel_keys,
            'start_key': start_key,
            'end_key': end_key,
            'gloss_key': gloss_key,
            'two_hand_map': two_hand_map,
            'channel_combine_map': channel_combine_map,
            'mask_key': mask_key,
        }
        self.gram_params = {
            'time_n': time_n,
            'channel_n': channel_n,
            'channels': channels,
            'method': method,
            'swap_map': swap_map,
            'hand_channels': hand_channels,
        }
        self.signbleu_params = {
            'smoothing': smoothing,
            'effective_order': effective_order,
            'verbose': verbose,
        }

    def _combine_params(
        self,
        default_params: Dict[str, Any],
        params: Dict[str, Any],
        ignore: Tuple[str] = ('self',),
    ):
        default_params = copy.deepcopy(default_params)
        for k, v in params.items():
            if k in ignore:
                continue
            if v is not None:
                default_params[k] = v
        return default_params

    def _block(
        self,
        data: List[Dict[str, List]],
        offset_threshold: Optional[float] = None,
        channel_keys: Optional[Tuple[str]] = None,
        start_key: Optional[str] = None,
        end_key: Optional[str] = None,
        gloss_key: Optional[str] = None,
        two_hand_map: Optional[Dict[str, List[str]]] = None,
        channel_combine_map: Optional[Dict[str, str]] = None,
        mask_key: Optional[str] = None,
    ):
        params = self._combine_params(
            self.block_params,
            locals(),
            ignore=('data', 'self'),
        )
        return [dict_to_block(datum, **params) for datum in data]

    def _get_lengths(self, blocks, ref_blocks):
        pred_lengths = [
            count_glosses_from_blocks(block, channels=None)
            for block in blocks
        ]
        ref_lengths = list()
        for pred_i in range(len(blocks)):
            pred_i_ref_lengths = [
                count_glosses_from_blocks(block_group[pred_i], channels=None)
                for block_group in ref_blocks
                if block_group[pred_i] is not None
            ]
            length = get_similar_length(
                pred_lengths[pred_i],
                pred_i_ref_lengths,
            )
            ref_lengths.append(length)
        return pred_lengths, ref_lengths

    def _gram(
        self,
        data,
        time_n=None,
        channel_n=None,
        channels=None,
        method=None,
        swap_map=None,
        hand_channels=None,
    ):
        params = self._combine_params(
            self.gram_params,
            locals(),
            ignore=('data', 'self'),
        )
        return [block_to_gram(datum, **params) for datum in data]

    def _signbleu(
        self,
        preds: List[Dict[str, List[Gram]]],
        refs: List[List[Dict[str, List[Gram]]]],
        pred_lengths: Optional[List[int]] = None,
        ref_lengths: Optional[List[int]] = None,
        smoothing = None,
        effective_order: bool = None,
        verbose = None,
    ):
        params = self._combine_params(
            self.signbleu_params,
            locals(),
            ignore=('preds', 'refs', 'pred_lengths', 'ref_lengths', 'self'),
        )
        return signbleu(preds, refs, pred_lengths, ref_lengths, **params)

    def _build_output(
        self,
        scores,
        hypothesis_format,
        reference_format,
    ):
        sig_params = {
            'metric': self.__class__.__name__,
            'scores': scores,
            'gram_params': self.gram_params,
            'metric_params': self.signbleu_params,
        }
        if hypothesis_format == 'neutral':
            sig_params['hypothesis_block_params'] = self.block_params
        if reference_format == 'neutral':
            sig_params['reference_block_params'] = self.block_params
        signature = Signature(**sig_params)
        output = Output(
            signature=signature,
            data=scores,
        )
        return output

    def calculate(
        self,
        hypotheses: List[Dict[str, List]],
        references: List[List],
        hypothesis_format: str = 'neutral',
        reference_format: str = 'neutral',
    ):
        r"""
        Calculate SignBLEU.

        Args:
            hypotheses (list[dict[str, list]]): Hypothesis data as a list of
                dicts where each dict is a time-aligned annotation hypothesis.
            references (list[list[dict[str, list] or None]]): Reference data as
                multiple lists of annotation references. The first list must
                contain one reference for each hypothesis. Subsequent lists can
                contain either a reference or None for each hypothesis.
            hypothesis_format (str, optional): Identifier specifying the
                hypothesis data format. Available options are "neutral" and
                "block".
            reference_format (str, optional): .

        """
        assert hypothesis_format in ['neutral', 'block']
        assert reference_format in ['neutral', 'block']

        if hypothesis_format == 'neutral':
            blocks = self._block(hypotheses)
        if reference_format == 'neutral':
            ref_blocks = [self._block(ref) for ref in references]

        lengths, ref_lengths = self._get_lengths(blocks, ref_blocks)

        grams = self._gram(blocks)
        ref_grams = [self._gram(block) for block in ref_blocks]
        scores = self._signbleu(
            grams,
            ref_grams,
            lengths,
            ref_lengths,
        )
        output = self._build_output(
            scores=scores,
            hypothesis_format=hypothesis_format,
            reference_format=reference_format,
        )
        return output
