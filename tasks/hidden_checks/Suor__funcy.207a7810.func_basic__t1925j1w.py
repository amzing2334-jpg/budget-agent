"""func_partial prepends every partial argument, and call keywords override partial keywords."""
from funcy.funcs import func_partial

f = func_partial(lambda *args, **kwargs: (args, kwargs), 1, 2, k=1)
assert f(3) == ((1, 2, 3), {"k": 1})
assert f(3, k=2) == ((1, 2, 3), {"k": 2})
