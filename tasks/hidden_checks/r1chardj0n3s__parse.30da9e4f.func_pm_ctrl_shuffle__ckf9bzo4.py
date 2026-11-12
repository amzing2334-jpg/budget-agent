"""Equal offsets compare equal; comparing with other objects never raises."""
from datetime import timezone

from parse import FixedTzOffset

a, b = FixedTzOffset(60, "+01:00"), FixedTzOffset(60, "+01:00")
assert a == b and not (a != b)
assert FixedTzOffset(30, "+00:30") != a
assert FixedTzOffset(0, "UTC") != timezone.utc
assert (a == "+01:00") is False and (a == object()) is False
