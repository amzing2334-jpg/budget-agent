import unittest
class Tests(unittest.TestCase):
    def test_case(self):
        with self.subTest(outer=True):
            with self.subTest(part='pass'): self.assertTrue(True)
            with self.subTest(part='skip'): self.skipTest('optional')
