import pytest
from pprint import pprint
from copy import deepcopy

from signbleu.outputs import (
    Output,
    Signature,
)

from utils import catch


class TestSignature:
    @pytest.mark.parametrize(
        'params',
        (
            {},
            {'metric': 'signbleu'},
        )
    )
    def test_init_success(self, params):
        signature = Signature(**params)

    @pytest.mark.parametrize(
        'params,target_signature',
        (
            ({}, 'off:na||t:na|c:na|dim:na||m:na|sm:na|eff:na||v:0.1.0'),
            ({'metric': 'signbleu'}, 'off:na||t:na|c:na|dim:na||m:sbleu|sm:na|eff:na||v:0.1.0'),
            (
                {'metric': 'signbleu', 'gram_params': {'time_n': 3, 'channel_n': 3}},
                'off:na||t:3|c:3|dim:na||m:sbleu|sm:na|eff:na||v:0.1.0',
            ),
        )
    )
    def test_signature(self, params, target_signature):
        signature = Signature(**params)
        with catch():
            assert str(signature) == target_signature


class TestOutput:
    @pytest.mark.parametrize(
        'signature_params,data',
        (
            ({'metric': 'signbleu'}, {'score': 50.0}),
        )
    )
    def test_init_success(self, signature_params, data):
        signature = Signature(**signature_params)
        output = Output(
            signature,
            data
        )

    @pytest.mark.parametrize(
        'signature_params,data',
        (
            ({'metric': 'signbleu'}, {}),
            ({'metric': 'signbleu'}, tuple()),
        )
    )
    def test_init_error(self, signature_params, data):
        signature = Signature(**signature_params)
        with pytest.raises(AssertionError):
            output = Output(signature, data)

    @pytest.mark.parametrize(
        'signature_params,data,target_signature',
        (
            # no parameters
            ({'metric': 'signbleu'}, {'score': 50.0}, 'off:na||t:na|c:na|dim:na||m:sbleu|sm:na|eff:na||v:0.1.0'),
            (
                {'metric': 'signbleu', 'gram_params': {'time_n': 3, 'channel_n': 3, 'method': '1d'}},
                {'score': 50.0},
                'off:na||t:3|c:3|dim:1||m:sbleu|sm:na|eff:na||v:0.1.0',
            ),
        )
    )
    def test_signature(self, signature_params, data, target_signature):
        signature = Signature(**signature_params)
        output = Output(
            signature,
            data
        )
        with catch():
            assert output.signature == target_signature
