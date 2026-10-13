import unittest

EXPECTED_COUNT = 42


def value():
    return 41 + 1


class TestTieBreak(unittest.TestCase):
    # Both conventions are used twice: the framework default ("actual first") wins the tie
    def test_tie(self):
        self.assertEqual(value(), 42)
        self.assertEqual(value(), EXPECTED_COUNT)

        self.assertEqual(42, value())
#       Noncompliant@-1 {{Swap these 2 arguments so the actual value comes first, to match the rest of this file.}}
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^@-1
#                        ^^@-2< {{Expected value.}}
#                            ^^^^^^^@-3< {{Actual value.}}
        self.assertEqual(EXPECTED_COUNT, value())  # Noncompliant
