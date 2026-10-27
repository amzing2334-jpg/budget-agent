import unittest
class Tests(unittest.TestCase):
    def test_case(self):
        with self.subTest(part='failure'): self.assertTrue(False)
        with self.subTest(part='skip'): self.skipTest('optional')
