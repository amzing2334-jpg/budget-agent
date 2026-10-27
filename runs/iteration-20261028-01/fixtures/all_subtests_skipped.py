import unittest
class Tests(unittest.TestCase):
    def test_case(self):
        for i in range(2):
            with self.subTest(i=i): self.skipTest('optional')
