import unittest
from slugify import slugify


class SlugTests(unittest.TestCase):
    def test_words(self):
        self.assertEqual(slugify(" Hello World "), "hello-world")

    def test_punctuation(self):
        self.assertEqual(slugify("one...two"), "one-two")
