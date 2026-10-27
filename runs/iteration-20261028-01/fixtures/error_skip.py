import unittest
class Tests(unittest.TestCase):
    def test_case(self):
        with self.subTest(part='error'): raise RuntimeError('test error')
        with self.subTest(part='skip'): self.skipTest('optional')
