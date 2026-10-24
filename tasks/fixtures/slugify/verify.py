import unittest
from slugify import slugify


class HiddenTests(unittest.TestCase):
    def test_digits(self):
        self.assertEqual(slugify("Release 2026 v2"), "release-2026-v2")
        self.assertEqual(slugify("123"), "123")

    def test_edges(self):
        self.assertEqual(slugify("__A9___B0__"), "a9-b0")
        self.assertEqual(slugify(""), "")
        self.assertEqual(slugify("!!!"), "")

    def test_regression(self):
        self.assertEqual(slugify(" Hello World "), "hello-world")


if __name__ == "__main__":
    unittest.main(verbosity=2)
