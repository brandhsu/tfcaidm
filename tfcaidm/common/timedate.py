"""Tools to get or measure time"""

import time
import pytz
import datetime
from pytz import timezone

import tfcaidm.common.constants as constants


def get_date():
    date = datetime.datetime.now(tz=pytz.utc)
    date = date.astimezone(timezone("US/Pacific"))
    return date.strftime(constants.date_format)


def get_mdy():
    return datetime.strptime(get_date(), constants.date_format).strftime("%b %d, %Y")


def timediff(end, start):
    """Measure the time difference between two datetime objects

    Args:
        end (datetime): latest datetime
        start (datetime): earliest datetime

    Returns:
        string: time difference as a string
    """

    end_date = datetime.datetime.strptime(end, constants.date_format)
    start_date = datetime.datetime.strptime(start, constants.date_format)

    return end_date - start_date


def profile(func):
    start = time.time()

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    end = time.time()
    print(end - start)

    return wrapper
