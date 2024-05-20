import logging


logger = logging.getLogger(__name__)


import os
import sys
import json
import time
import requests
import datetime
from pathlib import Path
from collections import deque
from traceback import format_exc
from typing import Optional, Union
from itertools import chain, combinations
from sacrebleu.compat import (
    sentence_bleu,
    corpus_bleu,
    sentence_chrf,
    corpus_chrf,
)
from sacrebleu.metrics import BLEU, TER
import string as string_module
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


from signbleu.rwlock import RWLock


SERVER = {
    #'host': 'localhost',
    'host': '192.168.1.25',
    'port': 5679,
}


CONT_KEY = ':'


COMPRESS_CHAR_MAP = {
    i: c
    for i, c in enumerate(string_module.ascii_letters + string_module.digits)
}


def query(
        path: str,
        payload: Union[dict, str],
        host: Optional[str] = None,
        port: Optional[int] = None,
        method: str = 'GET',
        headers: dict = {'Content-Type': 'application/json'},
        ssl: bool = False,
        retry: int = 0,
        suppress_warnings: bool = False,
):
    known_paths = [
        'id_to_text',
        'id_to_block',
        'text_to_block',
        'nia_to_block',
        'linear_to_block',
        'sign_bleu',
        'sos',  # separate original strict SignBLEU
        'metrics',
        'test',  # temporary path used for testing
    ]
    assert path in known_paths, f"Unknown path {path}"
    if isinstance(payload, dict):
        payload = json.dumps(payload)
    if host is None:
        host = SERVER['host']
    if port is None:
        port = SERVER['port']
    if ssl:
        protocol = 'https'
    else:
        protocol = 'http'
    url = f"{protocol}://{host}:{port}/{path}"
    response = None
    while retry >= 0 and response is None:
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                data=payload,
            )
        except Exception as e:
            error = repr(e)
            retry -= 1
    if response is None:
        response = {'error': error}
    else:
        response = response.json()
    if suppress_warnings:
        return response
    warnings = response.get('warning', list())
    if len(warnings) > 0:
        warnings = list(set(response['warning']))
        for warning in warnings:
            logger.warn(warning)
    return response


def corpus_bleu_n(
        hypotheses,
        references,
        max_ngram_order=4,
        lowercase=False,
        force=False,
        tokenize='none',
        smooth_method='exp',
        smooth_value=None,
        effective_order=False,
):
    scorer = BLEU(
        lowercase=lowercase,
        force=force,
        tokenize=tokenize,
        smooth_method=smooth_method,
        smooth_value=smooth_value,
        effective_order=effective_order,
        max_ngram_order=max_ngram_order,
    )
    return scorer.corpus_score(hypotheses, references).score


def sentence_bleu_n(
        hypotheses,
        references,
        max_ngram_order=4,
        lowercase=False,
        force=False,
        tokenize='none',
        smooth_method='exp',
        smooth_value=None,
        effective_order=True,
):
    scorer = BLEU(
        lowercase=lowercase,
        force=force,
        tokenize=tokenize,
        smooth_method=smooth_method,
        smooth_value=smooth_value,
        effective_order=effective_order,
        max_ngram_order=max_ngram_order,
    )
    return scorer.sentence_score(hypotheses, references).score


def corpus_ter(
        hypotheses,
        references,
        normalized=False,
        no_punct=False,
        asian_support=False,
        case_sensitive=False,
):
    scorer = TER(
        normalized=normalized,
        no_punct=no_punct,
        asian_support=asian_support,
        case_sensitive=case_sensitive,
    )
    return scorer.corpus_score(hypotheses, references).score


def sentence_ter(
        hypotheses,
        references,
        normalized=False,
        no_punct=False,
        asian_support=False,
        case_sensitive=False,
):
    scorer = TER(
        normalized=normalized,
        no_punct=no_punct,
        asian_support=asian_support,
        case_sensitive=case_sensitive,
    )
    return scorer.sentence_score(hypotheses, references).score


def strip_gloss(gloss):
    # return gloss.replace('~1::', '').replace('~2::', '').replace('&1::', '').replace('&2::', '').replace('0::', '').replace('1::', '').replace('2::', '')
    gloss = gloss.replace('~1::', '')
    gloss = gloss.replace('~2::', '')
    gloss = gloss.replace('&1::', '')
    gloss = gloss.replace('&2::', '')
    gloss = gloss.replace('1::', '')
    gloss = gloss.replace('2::', '')
    if gloss.startswith('::'):
        gloss = gloss[2:]

    if gloss.startswith('$'):
        gloss = gloss[1:]
    if gloss[-1] == '*':
        gloss = gloss[:-1]
    if gloss[-1] == '^':
        gloss = gloss[:-1]
    return gloss


def reduce_to_suji(linear):
    #### Need to check if hands are removed too ####
    if linear == '':
        return linear
    linear = ' '.join([
        strip_gloss(gloss)
        for gloss in linear.split(' ')
        if not gloss.startswith('::') and gloss != ''
    ])
    return linear


def suji_bleu(linear0, *linear1, n=4, effective_order=True):
    r"""
    Suji-only BLEU (linearization -> remove non glosses -> BLEU)
    """
    linear0 = reduce_to_suji(linear0)
    linear1 = [reduce_to_suji(linear) for linear in linear1]
    #return sentence_bleu(linear0, linear1, tokenize="none").score
    return sentence_bleu_n(linear0, linear1, max_ngram_order=n, effective_order=effective_order)


def suji_ter(linear0, *linear1, **kwargs):
    r"""
    Suji-only TER (linearization -> remove non glosses -> TER)
    """
    linear0 = reduce_to_suji(linear0)
    linear1 = [reduce_to_suji(linear) for linear in linear1]
    #return sentence_bleu(linear0, linear1, tokenize="none").score
    return sentence_ter(linear0, linear1, **kwargs)


def corpus_suji_bleu(linears0, linears1, n=4):
    r"""
    Suji-only BLEU (linearization -> remove non glosses -> BLEU)
    """
    linears0 = [
        reduce_to_suji(linear)
        for linear in linears0
    ]
    linears1 = [
        [
            reduce_to_suji(linear) if linear is not None else None
            for linear in linear_set
        ]
        for linear_set in linears1
    ]
    #return corpus_bleu(linears0, linears1, tokenize="none").score
    return corpus_bleu_n(linears0, linears1, max_ngram_order=n)


def corpus_suji_ter(linears0, linears1, **kwargs):
    r"""
    Suji-only TER (linearization -> remove non glosses -> TER)
    """
    linears0 = [
        reduce_to_suji(linear)
        for linear in linears0
    ]
    linears1 = [
        [
            reduce_to_suji(linear) if linear is not None else None
            for linear in linear_set
        ]
        for linear_set in linears1
    ]
    #return corpus_bleu(linears0, linears1, tokenize="none").score
    return corpus_ter(linears0, linears1, **kwargs)


def block_to_linear(
        blocks,
        left='left',
        right='right',
        channels=('left', 'right', 'mouth', 'eyebrow', 'head'),
):
    # does not add signs for the both (0::) channel
    output = list()
    nms = list(set(channels) - set([left, right]))
    for block in blocks:
        for channel in [right, left]:
            if block.get(channel) is None:
                continue
            if channel == right and block[channel][0] != ':':
                gloss = '1::' + block[channel].replace(':', '')
                if block.get(left) is not None and block[left][0] == ':':
                    gloss = '~' + gloss
                output.append(gloss)
            if channel == left and block[channel][0] != ':':
                gloss = '2::' + block[channel].replace(':', '')
                if block.get(right) is not None and block[right][0] == ':':
                    gloss = '~' + gloss
                elif block.get(right) is not None and block[right][0] != ':':
                    gloss = '&' + gloss
                output.append(gloss)
            for nms_channel in nms:
                if block.get(nms_channel) is not None:
                    output.append('::' + block['mouth'].replace(':', '').replace(' ', '_'))
    return ' '.join(output)


def process_hands(linear, separate_hands):
    def is_handed(gloss):
        return gloss[:2] in ['~1', '&1', '~2', '&2', '0:', '1:', '2:', '::']
    def is_suji(gloss):
        try:
            re.match('(~|&)?([0-2]::)?[A-Za-z]+[0-9][0-9]?', gloss).group()
            return True
        except:
            return False
    def separate_hand(gloss):
        if not is_suji(gloss):
            return gloss
        return ':: '.join(gloss.split('::'))

    linear = [
        gloss if (is_handed(gloss) or not is_suji(gloss)) else '0::' + gloss
        for gloss in linear.split(' ')
    ]
    linear = [
        separate_hand(gloss) if separate_hands else gloss
        for gloss in linear
    ]
    return ' '.join(linear)


def linear_bleu(linear0, *linear1, separate_hands=False, n=4, effective_order=True):
    r"""
    BLEU calculated on the linearization (linearization -> add space between hand and gloss -> BLEU)

    Args:
        linear0
        *linear1
        separate_hands
        n (int, optional): the ngram order
        effective_order: If True, use only non-zero ngram scores
            (for order <= `n`\). If False, use the order `n`\.
    """
    linear0 = process_hands(linear0, separate_hands=separate_hands)
    linear1 = [
        process_hands(linear, separate_hands=separate_hands)
        for linear in linear1
    ]
    #return sentence_bleu(linear0, linear1, tokenize="none").score
    return sentence_bleu_n(linear0, linear1, max_ngram_order=n, effective_order=effective_order)


def linear_ter(linear0, *linear1, separate_hands=False, **kwargs):
    r"""
    """
    linear0 = process_hands(linear0, separate_hands=separate_hands)
    linear1 = [
        process_hands(linear, separate_hands=separate_hands)
        for linear in linear1
    ]
    #return sentence_bleu(linear0, linear1, tokenize="none").score
    return sentence_ter(linear0, linear1, **kwargs)


def linear_chrf(linear0, *linear1, separate_hands=False, word_order=0):
    r"""
    CHRF calculated on the linearization (linearization -> add space between hand and gloss -> CHRF)
    """
    linear0 = process_hands(linear0, separate_hands=separate_hands)
    linear1 = [
        process_hands(linear, separate_hands=separate_hands)
        for linear in linear1
    ]
    return sentence_chrf(linear0, linear1, word_order=word_order).score


def suji_chrf(linear0, *linear1, separate_hands=False, word_order=0):
    r"""
    CHRF calculated on the suji linearization (linearization -> suji -> add space between hand and gloss -> CHRF)
    """
    linear0 = reduce_to_suji(linear0)
    linear1 = [
        reduce_to_suji(linear)
        for linear in linear1
    ]
    return sentence_chrf(linear0, linear1, word_order=word_order).score


def corpus_linear_bleu(linears0, linears1, separate_hands=False, n=4):
    r"""
    BLEU calculated on the linearization (linearization -> add space between hand and gloss -> BLEU)
    """
    linears0 = [
        process_hands(linear, separate_hands=separate_hands)
        for linear in linears0
    ]
    linears1 = [
        [
            process_hands(linear, separate_hands=separate_hands) if linear is not None else None
            for linear in linear_set
        ]
        for linear_set in linears1
    ]
    #return corpus_bleu(linears0, linears1, tokenize="none").score
    return corpus_bleu_n(linears0, linears1, max_ngram_order=n)


def corpus_linear_ter(linears0, linears1, separate_hands=False, **kwargs):
    r"""
    """
    linears0 = [
        process_hands(linear, separate_hands=separate_hands)
        for linear in linears0
    ]
    linears1 = [
        [
            process_hands(linear, separate_hands=separate_hands) if linear is not None else None
            for linear in linear_set
        ]
        for linear_set in linears1
    ]
    return corpus_ter(linears0, linears1, **kwargs)


def corpus_linear_chrf(linears0, linears1, separate_hands=False, word_order=0):
    r"""
    CHRF calculated on the linearization (linearization -> add space between hand and gloss -> CHRF)
    """
    linears0 = [
        process_hands(linear, separate_hands=separate_hands)
        for linear in linears0
    ]
    linears1 = [
        [
            process_hands(linear, separate_hands=separate_hands) if linear is not None else None
            for linear in linear_set
        ]
        for linear_set in linears1
    ]
    return corpus_chrf(linears0, linears1, word_order=word_order).score


def corpus_suji_chrf(linears0, linears1, separate_hands=False, word_order=0):
    r"""
    CHRF calculated on the suji linearization (linearization -> remove NMS -> add space between hand and gloss -> CHRF)
    """
    linears0 = [
        reduce_to_suji(linear)
        for linear in linears0
    ]
    linears1 = [
        [
            reduce_to_suji(linear) if linear is not None else None
            for linear in linear_set
        ]
        for linear_set in linears1
    ]
    return corpus_chrf(linears0, linears1, word_order=word_order).score


def merge_consecutive_identical_blocks(blocks):
    def gloss_eq(b1, b2, c):
        if all(b.get(c) is None for b in [b1, b2]):
            return True
        elif any(b.get(c) is None for b in [b1, b2]):
            return False
        return b1[c].replace(':', '') == b2[c].replace(':', '')

    def gloss_cont(b1, b2, c):
        if all(b.get(c) is None for b in [b1, b2]):
            return True
        elif any(b.get(c) is None for b in [b1, b2]):
            return False
        return b1[c][-1] == b2[c][0] == ':'

    delete_inds = list()
    for block_i in range(len(blocks) - 1):
        block1 = blocks[block_i]
        block2 = blocks[block_i+1]
        channels = set(list(block1.keys()) + list(block2.keys()))
        gloss_match = all(
            gloss_eq(block1, block2, channel) and gloss_cont(block1, block2, channel)
            for channel in channels
        )
        if gloss_match:
            delete_inds.append(block_i + 1)
    for ind in delete_inds[::-1]:
        for channel in blocks[ind]:
            text1 = blocks[ind-1][channel]
            text2 = blocks[ind][channel]
            if text2 is None:
                continue
            if text2[-1] != ':':
                text1 = text1[:-1]
            blocks[ind-1][channel] = text1
        blocks.pop(ind)
    return blocks


def extract_manual_sub_blocks(blocks, manual_channels=('right', 'left')):
    blocks = [
        {
            channel: gloss
            for channel, gloss in block.items()
            if channel in manual_channels
        }
        for block in blocks
    ]
    blocks = [block for block in blocks if len(block) > 0]
    blocks = merge_consecutive_identical_blocks(blocks)
    return blocks


def has_value(d, k):
    r"""
    Check if the dict `d` has a value at key `k`\.
    """
    return d.get(k) is not None


def remove_continued(gloss):
    return gloss.replace(CONT_KEY, '')


def compress_to_text(value):
    v = int(value)
    xs = list()
    base = len(COMPRESS_CHAR_MAP)
    while v >= base:
        xs.append(v % base)
        v = v // base
    xs.append(v)
    xs = xs[::-1]
    ts = ''.join([COMPRESS_CHAR_MAP[x] for x in xs])
    return ts


class Present:
    r"""
    Dummy class that matches the API for a Futures object but with a current
    result.

    (Complete API TBD)
    """
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result

    def done(self):
        return True


class EmptyQueue:
    def __init__(self):
        pass

    def empty(self):
        return True

    def qsize(self):
        return 0


class SequentialExecutor:
    r"""
    Executes tasks sequentially with the same API as ThreadPoolExecutor and
    ProcessPoolExecutor.

    (Complete API TBD)
    """
    def __init__(self, max_workers=None):
        self._work_queue = EmptyQueue()
        self._call_queue = EmptyQueue()

    def submit(self, function, *args, **kwargs):
        return Present(function(*args, **kwargs))

    def shutdown(self, wait=None):
        return None


class Background:
    r"""
    Helper class for running tasks in the background.

    All tasks should be set with a dict and key for setting the result.
    Add tasks with the :meth:`set` method and ensure all set tasks are
    completed by calling the :meth:`resolve` method.

    Example:
        >>> from utils import Background
        >>> from time import sleep
        >>> def calculate(x):
        >>>     sleep(1)
        >>>     return x * 2
        >>> background = Background(max_workers=4, parallel='thread')
        >>> output = dict()
        >>> for i, datum in enumerate(range(3)):
        >>>     background.set(output, i, calculate, datum)
        >>> print(output)
        {}
        >>> background.resolve()
        >>> print(output)
        {0: 0, 1: 1, 2: 4}

    """
    def __init__(
            self,
            max_workers=4,
            parallel=None,  # None, 'thread', 'process', 'sequential'
            pbar=None,
            block_on_set=False,
            verbose=True,
    ):
        r"""
        Initialize :class:`Background`\.

        Args:
            max_workers (int, optional): Number of workers to use. Ignored if
                `parallel` is 'sequential'.
                Defaults to 4.
            parallel (str, optional): The type of parallelism (or lack thereof)
                to use for the tasks. Available options are 'thread',
                'process', and 'sequential'. If 'thread', uses
                ThreadPoolExecutor. If 'process', uses ProcessPoolExecutor. If
                'sequential', does not use concurrent evaluation.
                If None, attempts to parse based on the OS. If Linux, set to
                'process'. Otherwise, set to 'thread'.
                Defaults to None.
            pbar (tqdm.tqdm, optional): Optional progress bar. If given, will
                call its :meth:`update` method after finishing each task during
                the :meth:`resolve` call.
                Defaults to None.
            block_on_set (bool, optional): If True, calls to :meth:`set` method
                block until space is available in the process or thread pool.
                If False, all tasks are offloaded to the executor as soon as
                they are set.
                Defaults to False.
            verbose (bool, optional): Verbosity.
                Ignored if pbar is not None.
                Defaults to True.

        Note:
            The :class:`Background` API is consistent regardless of
            `parallel`\s value.

        Note:
            If not using the `pbar` argument but using tqdm in an external
            loop, setting `block_on_set` to True and `verbose` to False will
            provide better progress updates.
        """
        if parallel is None:
            if sys.platform.startswith('linux'):
                parallel = 'process'
            else:
                parallel = 'thread'
        assert parallel in ['thread', 'process', 'sequential']

        self.max_workers = max_workers
        if parallel == 'thread':
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        elif parallel == 'process':
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        elif parallel == 'sequential':
            self.executor = SequentialExecutor()
        else:
            raise NotImplementedError
        self.to_resolve = deque()
        self.parallel = parallel
        self.pbar = pbar
        self.verbose = verbose
        self.block_on_set = block_on_set
        self.lock = RWLock()

    def _set(self, function, *args, **kwargs):
        if self.parallel == 'process':
            queue = self.executor._call_queue
        else:
            queue = self.executor._work_queue
        ## change to check qsize < max_workers
        #while self.block_on_set and not queue.empty():
        while self.block_on_set and queue.qsize() >= self.max_workers:
            # without a lock and with a lot of bad luck (and with multiple
            # concurrent calls to _set, this could be stuck forever.
            time.sleep(0.05)
        return self.executor.submit(function, *args, **kwargs)

    def set(self, obj, key, function, *args, **kwargs):
        r"""
        Add task to background task lists.

        Args:
            obj (dict): Dictionary to hold result.
            key (hashable): Key to which result will be mapped.
            function (callable): Callable task.
            *args: args to pass to `function`\.
            **kwargs: kwargs to pass to `function`\.
        """
        future = self._set(function, *args, **kwargs)
        self.to_resolve.append({
            'obj': obj,
            'key': key,
            'future': future,
            'time': time.time()
        })

    #def resolve(self, verbose=False):
    #    #for block in self.to_resolve:
    #    while len(self.to_resolve) > 0:
    #        print(f'Processing {len(self.to_resolve)} instances...', end='\r')
    #        block = self.to_resolve.popleft()
    #        block['obj'][block['key']] = block['future'].result()
    def _format_time(self, start):
        est_time = len(self.to_resolve) * (time.time() - start)
        if self.parallel != 'sequential':
            est_time /= self.max_workers
        return str(datetime.timedelta(seconds=int(est_time)))

    def resolve(self):
        r"""
        Blocks until all tasks have been completed.
        """
        #for block in self.to_resolve:
        if self.verbose and self.pbar is None:
            print(f'Processing {len(self.to_resolve)} instances', end='\r')
        while len(self.to_resolve) > 0:
            block = self.to_resolve.popleft()
            result = block['future'].result()
            with self.lock.w_locked():
                block['obj'][block['key']] = result
            if self.verbose and self.pbar is None:
                est_time = self._format_time(block['time'])
                print(f'{len(self.to_resolve)} (est: {est_time})', end='\r')
            elif self.pbar is not None:
                self.pbar.update()

    def soft_resolve(self, wait=0):
        r"""
        Resolve finished tasks.

        Args:
            wait (int, optional): How many seconds to wait for each task to
                resolve.
                Defaults to 0.

        Note:
            Need to change to use pointers and clean up after rather than
            popping the whole list.
        """
        if len([item for item in self.to_resolve if item['future'].done()]) == 0:
            return 0
        if self.verbose and self.pbar is None:
            print(f'Processing {len(self.to_resolve)} instances', end='\r')
        c = 0
        n = len(self.to_resolve)
        resolved = 0
        while c < n and len(self.to_resolve) > 0:
            block = self.to_resolve.popleft()
            if block['future'].done():
                result = block['future'].result()
                with self.lock.w_locked():
                    block['obj'][block['key']] = result
                if self.verbose and self.pbar is None:
                    est_time = self._format_time(block['time'])
                    print(f'{len(self.to_resolve)} (est: {est_time})', end='\r')
                elif self.pbar is not None:
                    self.pbar.update()
                resolved += 1
            else:
                self.to_resolve.append(block)
            c += 1
        return resolved

    def __del__(self):
        self.executor.shutdown(wait=False)


def nia_is_ku(name):
    return str(name).split('_')[-1].startswith('KU')


def nia_get_instance_id(name):
    name = '_'.join(str(name).split('_')[-6:])
    return name.split('.')[0]


def nia_get_group_id(name):
    return str(name).split('_')[-3]


def _log_error(e, path):
    path = Path(path) / f'error_{compress_to_text(time.time())}.txt'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf8', newline='') as f:
        f.write(e.__repr__().replace('\\n', '\n'))


def log_error(func, path='logs/'):
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error = format_exc()
            _log_error(error, path)
            return {}
    return wrapped


def powerset(iterable, n, start=1):
    "powerset([1,2,3], 3) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    n = min(len(s), n)
    return chain.from_iterable(combinations(s, r) for r in range(start, n+1))


ARGS_STR = """
Args:
"""


NOTE_STR = """
Note:
"""


def add_start_docstrings(*docstr):
    r"""from transformers/utils/doc.py"""
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn
    return docstring_decorator


def add_start_init_docstrings(*docstr):
    r"""from transformers/utils/doc.py"""
    def docstring_decorator(fn):
        fn.__init__.__doc__ = "".join(docstr) + (fn.__init__.__doc__ if fn.__init__.__doc__ is not None else "")
        return fn
    return docstring_decorator


def add_end_docstrings(*docstr):
    r"""from transformers/utils/doc.py"""
    def docstring_decorator(fn):
        fn.__doc__ = (fn.__doc__ if fn.__doc__ is not None else "") + "".join(docstr)
        return fn
    return docstring_decorator


def add_end_init_docstrings(*docstr):
    r"""from transformers/utils/doc.py"""
    def docstring_decorator(fn):
        fn.__init__.__doc__ = (fn.__init__.__doc__ if fn.__init__.__doc__ is not None else "") + "".join(docstr)
        return fn
    return docstring_decorator
