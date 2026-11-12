"""int_convert keeps base=None for automatic detection and converts every base."""
import parse

assert parse.int_convert().base is None
assert parse.parse("{:d}", "0x1A")[0] == 26 and parse.parse("{:d}", "-12")[0] == -12
