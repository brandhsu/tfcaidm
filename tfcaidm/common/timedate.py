"""Tools to get or measure time"""

import time
from datetime import datetime


def get_date():
    return time.strftime("%Y-%m-%d_%H-%M-%S_%Z", time.localtime())


def get_mdy(seconds=None):
    return time.strftime("%b %d, %Y", time.localtime(seconds))


def timediff(end, start):
    """Measure the time difference between two datetime objects

    Args:
        end (datetime): latest datetime
        start (datetime): earliest datetime

    Returns:
        string: time difference as a string
    """

    end_date = datetime.strptime(end, "%Y-%m-%d_%H-%M-%S_%Z")
    start_date = datetime.strptime(start, "%Y-%m-%d_%H-%M-%S_%Z")

    return end_date - start_date


def profile(func):
    start = time.time()

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    end = time.time()
    print(end - start)

    return wrapper
