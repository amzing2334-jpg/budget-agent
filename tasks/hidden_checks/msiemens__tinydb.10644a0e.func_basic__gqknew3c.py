"""update_multiple updates the matching documents and deletes none, whatever their ids."""
from tinydb import TinyDB, where
from tinydb.storages import MemoryStorage

db = TinyDB(storage=MemoryStorage)
db.insert_multiple({"n": i} for i in range(12))          # ids 1 to 12, two of them with two digits
db.update_multiple([({"x": 1}, where("n") == 0), ({"x": 2}, where("n") == 11)])
assert len(db) == 12
assert db.get(doc_id=1)["x"] == 1 and db.get(doc_id=12)["x"] == 2
assert all(db.get(doc_id=i) is not None for i in range(1, 13))
