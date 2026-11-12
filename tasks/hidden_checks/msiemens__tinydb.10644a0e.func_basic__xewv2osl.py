"""FrozenDict rejects every write, including clear() and popitem(), which take no arguments."""
from tinydb.utils import FrozenDict

frozen = FrozenDict({"a": 1})
for write in (frozen.clear, frozen.popitem, lambda: frozen.__setitem__("b", 2),
              lambda: frozen.__delitem__("a"), lambda: frozen.setdefault("b", 2)):
    try:
        write()
    except TypeError:
        continue
    raise AssertionError(f"FrozenDict accepted a write: {write}")
assert dict(frozen) == {"a": 1}
