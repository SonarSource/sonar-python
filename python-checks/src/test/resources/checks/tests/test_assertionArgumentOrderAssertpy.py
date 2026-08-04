from assertpy import assert_that

EXPECTED_COUNT = 42


def value():
    return 41 + 1


# Consistent expected-first convention → no issue
def test_assertpy_assertions_expected_first():
    assert_that(42).is_equal_to(value())
    assert_that(EXPECTED_COUNT).is_equal_to(value())
    assert_that(42).described_as("count").is_equal_to(value())
    assert_that(EXPECTED_COUNT).described_as("count").snapshot("baseline").is_equal_to(value())
    assert_that(42).is_equal_to(EXPECTED_COUNT)
    assert_that(value()).is_equal_to(value())
    assert_that(value()).is_equal_to()
    assert_that().is_equal_to(value())
    assert_that(value()).is_not_equal_to(42)
    builder.is_equal_to(42)
    factory().is_equal_to(42)
    other_call(value()).is_equal_to(42)
