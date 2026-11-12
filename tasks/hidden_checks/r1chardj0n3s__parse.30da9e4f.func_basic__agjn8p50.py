"""search() is case-insensitive by default and case-sensitive on request."""
import parse

assert parse.search("hello {}", "say HELLO world") is not None
assert parse.search("hello {}", "say HELLO world", case_sensitive=True) is None
