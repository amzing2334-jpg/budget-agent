import unittest
from intervals import merge_intervals


class HiddenTests(unittest.TestCase):
    def test_touching(self):
        self.assertEqual(merge_intervals([[1, 2], [2, 3]]), [[1, 3]])
        self.assertEqual(merge_intervals([[3, 4], [1, 2], [2, 3]]), [[1, 4]])

    def test_containment_and_points(self):
        self.assertEqual(merge_intervals([[1, 5], [2, 2], [5, 6]]), [[1, 6]])
        self.assertEqual(merge_intervals([[1, 1], [1, 1]]), [[1, 1]])

    def test_disjoint_and_input_unchanged(self):
        data = [[4, 5], [1, 2]]
        self.assertEqual(merge_intervals(data), [[1, 2], [4, 5]])
        self.assertEqual(data, [[4, 5], [1, 2]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
