# This module was borrowed and modified from
# https://gist.github.com/tylerneylon/a7ff6017b7a1f9a506cf75aa23eacfd6
# Original author: Tyler Neylon at Unbox Research
# Licensed as: Public Domain


"""
A basic Read-Write Lock.

Code from Tyler Neylon at Unbox Research (released as public domain):
`https://gist.github.com/tylerneylon/a7ff6017b7a1f9a506cf75aa23eacfd6`
"""


from contextlib import contextmanager
from threading  import Lock


class RWLock(object):
    r"""
    RWLock class.

    This is meant to allow an object to be read from by
    multiple threads, but only written to by a single thread at a time. See:
    https://en.wikipedia.org/wiki/Readers%E2%80%93writer_lock

    Example:
        >>> from rwlock import RWLock
        >>> my_obj_rwlock = RWLock()
        >>> # When reading from my_obj:
        >>> with my_obj_rwlock.r_locked():
        >>>     do_read_only_things_with(my_obj)
        >>> # When writing to my_obj:
        >>> with my_obj_rwlock.w_locked():
        >>>     mutate(my_obj)
    """

    def __init__(self):
        r"""Initialize :class:`RWLock`\."""

        self.w_lock = Lock()
        self.num_r_lock = Lock()
        self.num_r = 0

    # ___________________________________________________________________
    # Reading methods.

    def r_acquire(self):
        self.num_r_lock.acquire()
        self.num_r += 1
        if self.num_r == 1:
            self.w_lock.acquire()
        self.num_r_lock.release()

    def r_release(self):
        assert self.num_r > 0
        self.num_r_lock.acquire()
        self.num_r -= 1
        if self.num_r == 0:
            self.w_lock.release()
        self.num_r_lock.release()

    @contextmanager
    def r_locked(self):
        r"""This method is designed to be used via the `with` statement."""
        try:
            self.r_acquire()
            yield
        finally:
            self.r_release()

    # ___________________________________________________________________
    # Writing methods.

    def w_acquire(self):
        self.w_lock.acquire()

    def w_release(self):
        self.w_lock.release()

    @contextmanager
    def w_locked(self):
        """This method is designed to be used via the `with` statement."""
        try:
            self.w_acquire()
            yield
        finally:
            self.w_release()
