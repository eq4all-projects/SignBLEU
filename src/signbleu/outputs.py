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
Classes related to the score outputs and signatures.
"""


import signbleu


def _assert_nonempty_dict(data, name):
    assert isinstance(data, dict) and len(data) > 0, f"{name} must be a nonempty dict"


class Signature:
    r"""
    Score signature class.

    :class:`Signature` is not intended to be used by the end user. Instead, it
    is used by :class:`Output` when outputing the formatted signature.
    """
    missing_key = 'NA'
    sep_key = '||'
    def __init__(
        self,
        metric=None,
        scores=None,
        hypothesis_block_params=None,
        reference_block_params=None,
        gram_params=None,
        metric_params=None,
    ):
        r"""
        Initialize :class:`Signature`\.
        """
        self._metric = metric
        self._scores = scores

        if hypothesis_block_params is not None:
            _assert_nonempty_dict(
                hypothesis_block_params,
                name='hypothesis block params',
            )
        self._hypothesis_block_params = hypothesis_block_params

        if reference_block_params is not None:
            _assert_nonempty_dict(
                reference_block_params,
                name='reference block params',
            )
        self._reference_block_params = reference_block_params

        if gram_params is not None:
            _assert_nonempty_dict(
                gram_params,
                name='gram params',
            )
        self._gram_params = gram_params

        if metric_params is not None:
            _assert_nonempty_dict(
                metric_params,
                name='metric params',
            )
        self._metric_params = metric_params

        self._input_params = {
            'metric': metric,
            'scores': scores,
            'hypothesis_block_params': hypothesis_block_params,
            'reference_block_params': reference_block_params,
            'gram_params': gram_params,
            'metric_params': metric_params,
        }

    def _format_block_signature(self, params):
        block_sig_base = 'off:{}'
        if params is None:
            return block_sig_base.format(self.missing_key)

        offset = params.get('offset_threshold')
        if offset is None:
            offset = self.missing_key

        return block_sig_base.format(offset)

    def _parse_gram_size(self, n):
        if n == self.missing_key or isinstance(n, int):
            return n
        assert isinstance(n, tuple) or isinstance(n, list)
        assert len(n) == 2
        return f'{n[0]}-{n[1]}'

    def _format_gram_signature(self, params):
        gram_sig_base = 't:{}|c:{}|dim:{}'
        if params is None:
            return gram_sig_base.format(
                self.missing_key,
                self.missing_key,
                self.missing_key,
            )
        dim_abbr = {
            self.missing_key: self.missing_key,
            '1d': 1,
            '2d': 2,
            'time': 't',
            'channel': 'c',
        }
        t = self._parse_gram_size(params.get('time_n', self.missing_key))
        c = self._parse_gram_size(params.get('channel_n', self.missing_key))
        dimension = dim_abbr[params.get('method', self.missing_key)]
        return gram_sig_base.format(
            t,
            c,
            dimension,
        )

    def _format_metric_signature(self, metric, params):
        metric_sig_base = 'm:{}|sm:{}|eff:{}'
        metric_abbr = {
            self.missing_key: self.missing_key,
            'signbleu': 'sbleu',
        }
        smoothing_abbr = {
            self.missing_key: self.missing_key,
            None: 'none',
            'none': 'none',
            'epsilon': 'eps',
            'add_k': 'k',
            'floor': 'fl',
            'exponential': 'exp',
        }
        effective_abbr = {
            self.missing_key: self.missing_key,
            True: 'Y',
            False: 'N',
        }

        if metric is None:
            metric = self.missing_key
        else:
            metric = metric_abbr.get(metric.lower())

        if params is None:
            smoothing = effective = self.missing_key
        else:
            smoothing = smoothing_abbr.get(params.get('smoothing', self.missing_key))
            effective = effective_abbr.get(params.get('effective_order', self.missing_key))

        return metric_sig_base.format(
            metric,
            smoothing,
            effective,
        )

    def _format_version_signature(self):
        return f'v:{signbleu.__version__}'

    def _format_signature(self):
        signature = '{block}{sep}{gram}{sep}{metric}{sep}{version}'.format(
            block=self._format_block_signature(self._hypothesis_block_params),
            gram=self._format_gram_signature(self._gram_params),
            metric=self._format_metric_signature(self._metric, self._metric_params),
            version=self._format_version_signature(),
            sep=self.sep_key,
        )
        return signature.lower()

    def __str__(self):
        return self._format_signature()

    def __repr__(self):
        return f'Signature({repr(self._input_params)})'


class Output(dict):
    r"""
    Score outputs class.
    """
    def __init__(self, signature, data):
        r"""
        Initialize :class:`Output`\.
        """
        _assert_nonempty_dict(data, name='data params')
        assert isinstance(signature, Signature)
        # assert 'score' in data
        super().__init__(**data)
        self._signature = signature

    @property
    def score(self):
        r"""
        Return the corpus SignBLEU score.
        """
        if 'corpus_signbleu' in self:
            return self['corpus_signbleu']
        raise ValueError()

    @property
    def scores(self):
        r"""
        Return all score information.
        """
        return dict(self)

    @property
    def signature(self):
        r"""Return the formatted signature."""
        return str(self._signature)

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise ValueError(f'Unrecognized key "{attr}".')

    def __str__(self):
        key, value = list(self.items())[0]
        value = str(value)
        if len(value) > 7:
            value = value[:4] + '...'
        return f'Output({str(self._signature)}, {{{key}: {value}, ...}})'

    def __repr__(self):
        return f'Output({repr(self._signature)}, {super().__repr__()})'


class OutputConstructor:
    r"""
    Convenience class for constructing outputs
    """
    def __init__(self, signature_params):
        self._signature = Signature(signature_params)

    def construct_outputs(self, output_data):
        return Output(
            signature=self._signature,
            data=output_data,
        )
