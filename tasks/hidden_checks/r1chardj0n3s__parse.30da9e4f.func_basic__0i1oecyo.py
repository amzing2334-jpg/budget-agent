"""named_fields returns a copy: changing it does not change the parser."""
import parse

parser = parse.compile("{a} {b}")
fields = parser.named_fields
fields.append("x")
assert parser.named_fields == ["a", "b"]
