import unittest
from intervals import merge_intervals


class IntervalTests(unittest.TestCase):
    def test_overlap(self):
        self.assertEqual(merge_intervals([[1, 4], [2, 5]]), [[1, 5]])

    def test_empty(self):
        self.assertEqual(merge_intervals([]), [])
