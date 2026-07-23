import unittest


def test_and_noncompliant(user):
    assert user.is_active and user.is_verified  # Noncompliant {{Split this composite assertion into separate assertions.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    assert user.a and user.b and user.c  # Noncompliant
    assert (user.is_active and user.is_verified)  # Noncompliant

def test_not_or_noncompliant(axis):
    assert not (axis.visible or axis.label_visible)  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    assert not (axis.a or axis.b or axis.c)  # Noncompliant


def test_compliant(user, axis, a, b):
    assert user.is_active
    assert user.is_verified
    assert not axis.visible
    assert not axis.label_visible
    assert a or b  # plain or is compliant
    assert not a  # negated single value is compliant
    assert not (a and b)  # negated and is compliant
    assert a and b or b  # outer is or, compliant


def helper_function(a, b):
    assert a and b  # Noncompliant


class TestCompositeAssertion(unittest.TestCase):
    def test_and_noncompliant(self):
        assert self.a and self.b  # Noncompliant

    def test_not_or_noncompliant(self):
        assert not (self.x or self.y)  # Noncompliant

    def test_compliant(self):
        assert self.a
        assert self.b
        assert self.a or self.b
