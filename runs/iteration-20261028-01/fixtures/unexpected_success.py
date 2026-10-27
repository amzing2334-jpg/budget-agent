import unittest
class Tests(unittest.TestCase):
    @unittest.expectedFailure
    def test_case(self): self.assertTrue(True)
