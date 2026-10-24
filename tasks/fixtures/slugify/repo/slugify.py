"""Small URL slug helper."""
import re


def slugify(text):
    return re.sub(r"[^a-z]+", "-", text.lower()).strip("-")
