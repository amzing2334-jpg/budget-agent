"""one() is lazy (it stops after a second match) and accepts funcy's extended predicates."""
from itertools import count

from funcy import one

assert one(lambda x: x == 3, [1, 2, 3]) is True
assert one(lambda x: x > 0, count(1)) is False      # an infinite iterable with two matches
assert one(None, [0, 1]) is True                     # None means truthiness
