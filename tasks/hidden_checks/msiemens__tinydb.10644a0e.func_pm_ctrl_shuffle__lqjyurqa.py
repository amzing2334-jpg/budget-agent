"""Query.search matches strings only; other values never match."""
from tinydb import Query

query = Query().f.search("1")
assert query({"f": "a1"}) is True
assert query({"f": 1}) is False
