"""without() and lwithout() both skip the given items."""
from funcy import lwithout, without

assert list(without([1, 2, 3, 2], 2)) == [1, 3]
assert lwithout([1, 2, 3, 2], 2) == [1, 3]
