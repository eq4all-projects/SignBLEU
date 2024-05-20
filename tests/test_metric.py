r"""
Test SignBLEU calculation with :func:`signbleu` and :class:`SignBLEU`\.

TODO:
- confirm all calculations in this test module.
- Fix warnings when gram matches are 0
"""


import math
import pytest
from pprint import pprint
from copy import deepcopy

from signbleu import __version__ as version
from signbleu.block import (
    dict_to_block,
)
from signbleu.gram import (
    block_to_gram,
)
from signbleu.metric import (
    signbleu,
    count_glosses_from_blocks,
    SignBLEU,
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
        {'gloss': 'oh', 'start': 0.099, 'end': 0.8},
        {'gloss': 'there', 'start': 1.1, 'end': 1.7},
        {'gloss': 'you', 'start': 2.8, 'end': 4.1},
    ],
    'both': [
        {'gloss': 'friend', 'start': 1.95, 'end': 2.7},
    ],
}
similar_data = {
    'face1': [{'gloss': 'happy', 'start': 0.099, 'end': 0.3}],
    'face2': [{'gloss': '::excited::', 'start': 0.3, 'end': 0.8}],
    'left': [
        {'gloss': 'hello', 'start': 0.1, 'end': 0.7},
        {'gloss': 'are', 'start': 3.5, 'end': 4.1},
    ],
    'right': [
        # {'gloss': 'oh', 'start': 0.099, 'end': 0.8},
        {'gloss': 'there', 'start': 1.1, 'end': 1.7},
        {'gloss': 'you', 'start': 2.8, 'end': 4.1},
    ],
    'both': [
        {'gloss': 'friend', 'start': 1.95, 'end': 2.7},
    ],
}
semi_similar_data = {
    'face1': [{'gloss': 'sad', 'start': 0.099, 'end': 0.3}],
    'face2': [{'gloss': '::excited::', 'start': 0.3, 'end': 0.8}],
    'left': [
        {'gloss': 'hi', 'start': 0.1, 'end': 0.7},
        {'gloss': 'are', 'start': 3.5, 'end': 4.1},
    ],
    'right': [
        # {'gloss': 'oh', 'start': 0.099, 'end': 0.8},
        {'gloss': 'there', 'start': 1.1, 'end': 1.7},
        # {'gloss': 'you', 'start': 2.8, 'end': 4.1},
    ],
    'both': [
        {'gloss': 'friend', 'start': 1.95, 'end': 2.7},
    ],
}
different_data = {
    'face1': [{'gloss': 'sad', 'start': 0.099, 'end': 0.3}],
    'face2': [{'gloss': '::disappointed::', 'start': 0.3, 'end': 0.8}],
    'left': [
        {'gloss': 'oh', 'start': 0.1, 'end': 0.7},
        {'gloss': 'what', 'start': 3.5, 'end': 4.2},
    ],
    'right': [
        {'gloss': 'a', 'start': 1.3, 'end': 1.7},
        {'gloss': 'random', 'start': 2.8, 'end': 4.1},
    ],
    'both': [
        {'gloss': 'thing', 'start': 2.95, 'end': 2.7},
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


class TestSignBLEU:
    @pytest.mark.parametrize(
        'data,references,block_params,gram_params,signbleu_params,target_corpus_score,target_instance_score',
        (
            (
                data,
                (data, data),
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {'time_n': 3, 'channel_n': 2, 'channels': ['face', 'left', 'right']},
                {},
                1.0,
                1.0,
            ),
            (
                data,
                (data, similar_data),
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {'time_n': 3, 'channel_n': 2, 'channels': ['face', 'left', 'right']},
                {},
                1.0,
                1.0,
            ),
            (
                data,
                (similar_data, similar_data),
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {'time_n': 3, 'channel_n': 3, 'channels': ['face', 'left', 'right']},
                {},
                0.0,
                0.5479173,
            ),
            (
                data,
                (data, similar_data),
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {'time_n': 5, 'channel_n': 2, 'channels': ['face', 'left', 'right']},
                {},
                0.0,
                1.0,
            ),
        ),
    )
    def test_default_single(
        self,
        data,
        references,
        block_params,
        gram_params,
        signbleu_params,
        target_corpus_score,
        target_instance_score,
    ):
        blocks = dict_to_block(data, **block_params)
        ref_blocks = [
            dict_to_block(reference, **block_params)
            for reference in references
        ]
        grams = block_to_gram(blocks, **gram_params)
        ref_grams = [
            block_to_gram(ref_block, **gram_params)
            for ref_block in ref_blocks
        ]
        lengths = count_glosses_from_blocks(blocks, channels=None)
        # for this test, assume the first reference is the closer reference
        ref_lengths = count_glosses_from_blocks(ref_blocks[0], channels=None)
        score = signbleu(
            hypotheses=[grams],
            references=[[ref_gram_set] for ref_gram_set in ref_grams],
            hyp_lengths=[lengths],
            ref_lengths=[ref_lengths],
            **signbleu_params,
        )
        with catch():
            assert math.isclose(score['corpus_signbleu'], target_corpus_score, abs_tol=1e-06)
            assert math.isclose(score['instance_signbleu'][0], target_instance_score, abs_tol=1e-06)

    @pytest.mark.parametrize(
        'data,references,block_params,gram_params,signbleu_params,target_corpus_score,target_instance_scores',
        (
            (
                (data, similar_data),
                ((data, similar_data), (similar_data, data)),  # ((hyp1_ref1, hyp2_ref1), (hyp1_ref2, hyp2_ref2))
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {'time_n': 3, 'channel_n': 2, 'channels': ['face', 'left', 'right']},
                {},
                1.0,
                (1.0, 1.0),
            ),
            (
                (data, similar_data),
                ((similar_data, data), (different_data, different_data)),
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {'time_n': 3, 'channel_n': 2, 'channels': ['face', 'left', 'right']},
                {},
                0.790829,
                (2/3, 0.882497),
            ),
        ),
    )
    def test_default_multiple(
        self,
        data,
        references,
        block_params,
        gram_params,
        signbleu_params,
        target_corpus_score,
        target_instance_scores,
    ):
        blocks = [
            dict_to_block(datum, **block_params)
            for datum in data
        ]
        ref_blocks = [
            [dict_to_block(ref, **block_params) for ref in ref_set]
            for ref_set in references
        ]
        grams = [
            block_to_gram(block, **gram_params)
            for block in blocks
        ]
        ref_grams = [
            [block_to_gram(ref_block, **gram_params) for ref_block in ref_block_set]
            for ref_block_set in ref_blocks
        ]
        lengths = [
            count_glosses_from_blocks(block, channels=None)
            for block in blocks
        ]
        # for test, assume the first set of references is most similar
        ref_lengths = [
            count_glosses_from_blocks(ref_block, channels=None)
            for ref_block in ref_blocks[0]
        ]
        score = signbleu(
            hypotheses=grams,
            references=ref_grams,
            hyp_lengths=lengths,
            ref_lengths=ref_lengths,
            **signbleu_params,
        )
        with catch():
            assert math.isclose(score['corpus_signbleu'], target_corpus_score, abs_tol=1e-06)
            assert math.isclose(score['instance_signbleu'][0], target_instance_scores[0], abs_tol=1e-06)
            assert math.isclose(score['instance_signbleu'][1], target_instance_scores[1], abs_tol=1e-06)

    @pytest.mark.parametrize(
        'data,references,block_params,gram_params,signbleu_params,target_corpus_score,target_instance_scores',
        (
            (
                (data, similar_data),
                ((similar_data, data), (different_data, different_data)),
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {'time_n': 4, 'channel_n': 2, 'channels': ['face', 'left', 'right']},
                {'smoothing': 'epsilon'},
                0.0,
                (0.456170, 0.8824969),
            ),
            (
                (data, similar_data),
                ((similar_data, data), (different_data, different_data)),
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {'time_n': 3, 'channel_n': 2, 'channels': ['face', 'left', 'right']},
                {'smoothing': 'add_k'},
                0.790829,
                (0.713896, 0.882497),
            ),
            (
                (data, similar_data),
                ((similar_data, data), (different_data, different_data)),
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {'time_n': 4, 'channel_n': 2, 'channels': ['face', 'left', 'right']},
                {'smoothing': 'exponential'},
                0.0,
                (0.629392, 0.8824969),
            ),
            (
                (data, similar_data),
                ((similar_data, semi_similar_data), (different_data, different_data)),
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {'time_n': 3, 'channel_n': 3, 'channels': ['face', 'left', 'right']},
                {'smoothing': 'floor'},
                0.0,
                (0.397119, 0.236435),
            ),
            (
                (data, data),
                ((different_data, semi_similar_data), (different_data, different_data)),
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {'time_n': 3, 'channel_n': 3, 'channels': ['face', 'left', 'right']},
                {'smoothing': 'floor'},
                0.0,
                (0.0, 0.125283),
            ),
        ),
    )
    def test_smoothing(
        self,
        data,
        references,
        block_params,
        gram_params,
        signbleu_params,
        target_corpus_score,
        target_instance_scores,
    ):
        blocks = [
            dict_to_block(datum, **block_params)
            for datum in data
        ]
        ref_blocks = [
            [dict_to_block(ref, **block_params) for ref in ref_set]
            for ref_set in references
        ]
        grams = [
            block_to_gram(block, **gram_params)
            for block in blocks
        ]
        ref_grams = [
            [block_to_gram(ref_block, **gram_params) for ref_block in ref_block_set]
            for ref_block_set in ref_blocks
        ]
        lengths = [
            count_glosses_from_blocks(block, channels=None)
            for block in blocks
        ]
        # for test, assume the first set of references is most similar
        ref_lengths = [
            count_glosses_from_blocks(ref_block, channels=None)
            for ref_block in ref_blocks[0]
        ]
        score = signbleu(
            hypotheses=grams,
            references=ref_grams,
            hyp_lengths=lengths,
            ref_lengths=ref_lengths,
            **signbleu_params,
        )
        with catch():
            assert math.isclose(score['corpus_signbleu'], target_corpus_score, abs_tol=1e-06)
            assert math.isclose(score['instance_signbleu'][0], target_instance_scores[0], abs_tol=1e-06)
            assert math.isclose(score['instance_signbleu'][1], target_instance_scores[1], abs_tol=1e-06)

    @pytest.mark.parametrize(
        'data,references,block_params,gram_params,signbleu_params,target_corpus_score,target_instance_scores,signature',
        (
            (
                (data, similar_data),
                ((data, similar_data), (similar_data, data)),  # ((hyp1_ref1, hyp2_ref1), (hyp1_ref2, hyp2_ref2))
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {'time_n': 3, 'channel_n': 2, 'channels': ['face', 'left', 'right']},
                {},
                1.0,
                (1.0, 1.0),
                f'off:na||t:3|c:2|dim:1||m:sbleu|sm:exp|eff:y||v:{version}',
            ),
            (
                (data, similar_data),
                ((similar_data, data), (different_data, different_data)),
                {'offset_threshold': 0.01, 'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {'time_n': 1, 'channel_n': 2, 'channels': ['face', 'left', 'right']},
                {},
                0.761042,
                (0.628539, 0.882497),
                f'off:0.01||t:1|c:2|dim:1||m:sbleu|sm:exp|eff:y||v:{version}',
            ),
            (
                (data, similar_data),
                ((similar_data, data), (different_data, different_data)),
                {'channel_combine_map': channel_combine_map, 'two_hand_map': two_hand_map},
                {'time_n': 3, 'channel_n': 2, 'channels': ['face', 'left', 'right']},
                {},
                0.790829,
                (2/3, 0.882497),
                f'off:na||t:3|c:2|dim:1||m:sbleu|sm:exp|eff:y||v:{version}',
            ),
        ),
    )
    def test_signbleu_class(
        self,
        data,
        references,
        block_params,
        gram_params,
        signbleu_params,
        target_corpus_score,
        target_instance_scores,
        signature,
    ):
        SIGNBLEU = SignBLEU(**block_params, **gram_params, **signbleu_params)
        score = SIGNBLEU.calculate(data, references)
        with catch():
            assert score.signature == signature
            assert math.isclose(score.corpus_signbleu, target_corpus_score, abs_tol=1e-06)
            assert math.isclose(score.instance_signbleu[0], target_instance_scores[0], abs_tol=1e-06)
            assert math.isclose(score.instance_signbleu[1], target_instance_scores[1], abs_tol=1e-06)
