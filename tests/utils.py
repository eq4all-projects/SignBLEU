import os
from contextlib import (
    AbstractContextManager,
    ContextDecorator,
    contextmanager,
)


PYTEST_CATCH = os.environ.get('PYTEST_CATCH', False)


@contextmanager
def catch():
    try:
        yield None
    except Exception as e:
        if PYTEST_CATCH:
            breakpoint()
            message = (
                "Exception caught! Dropping into debugger. Go up two levels "
                "(`u<enter>u<enter>`) to see the code that threw the error."
            )
            print(message)
            print(e)
        else:
            raise e
