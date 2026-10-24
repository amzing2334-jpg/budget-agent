import unittest
from chunks import chunks


class ChunkTests(unittest.TestCase):
    def test_chunks(self):
        self.assertEqual(chunks([1, 2, 3], 2), [[1, 2], [3]])

    def test_empty(self):
        self.assertEqual(chunks([], 2), [])
