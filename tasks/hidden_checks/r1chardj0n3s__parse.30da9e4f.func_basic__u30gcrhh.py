"""A fixed offset is the same for every datetime, any date, and for time objects (dt=None)."""
from datetime import datetime, time, timedelta

from parse import FixedTzOffset

tz = FixedTzOffset(60, "+01:00")
hour = timedelta(minutes=60)
assert tz.utcoffset(None) == hour
assert tz.utcoffset(datetime(2020, 12, 31, 12, 0, tzinfo=tz)) == hour
assert tz.utcoffset(datetime(2020, 6, 1, tzinfo=tz)) == hour
assert time(10, 0, tzinfo=tz).utcoffset() == hour
