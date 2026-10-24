import unittest
from chunks import chunks


class HiddenTests(unittest.TestCase):
    def test_positive_size_required(self):
        for items in ([], [1, 2, 3]):
            for size in (0, -1, -10):
                with self.subTest(items=items, size=size):
                    with self.assertRaises(ValueError):
                        chunks(items, size)

    def test_valid_sizes(self):
        self.assertEqual(chunks([1, 2, 3], 1), [[1], [2], [3]])
        self.assertEqual(chunks([1, 2, 3], 2), [[1, 2], [3]])
        self.assertEqual(chunks([1, 2], 10), [[1, 2]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
