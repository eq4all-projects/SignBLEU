import logging


logger = logging.getLogger(__name__)


import sys
import time
import datetime
from pathlib import Path
from collections import deque
from traceback import format_exc
from typing import Optional
from itertools import chain, combinations
import string as string_module
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


from signbleu.rwlock import RWLock


COMPRESS_CHAR_MAP = {
    i: c
    for i, c in enumerate(string_module.ascii_letters + string_module.digits)
}


def has_value(d, k):
    r"""
    Check if the dict `d` has a value at key `k`\.
    """
    return d.get(k) is not None


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
