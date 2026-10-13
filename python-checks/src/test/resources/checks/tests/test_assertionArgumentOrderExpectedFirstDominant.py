from assertpy import assert_that

EXPECTED_COUNT = 42


def value():
    return 41 + 1


# "expected first" is used 4 times and "actual first" twice, so the latter deviates from the file convention
def test_assertpy_assertions():
    assert_that(42).is_equal_to(value())
    assert_that(EXPECTED_COUNT).is_equal_to(value())
    assert_that(42).described_as("count").is_equal_to(value())
    assert_that(EXPECTED_COUNT).described_as("count").snapshot("baseline").is_equal_to(value())

    assert_that(value()).is_equal_to(42)
#   Noncompliant@-1 {{Swap these 2 values so the expected value comes first, to match the rest of this file.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^@-1
#                                    ^^@-2< {{Expected value.}}
#               ^^^^^^^@-3< {{Actual value.}}
    assert_that(value()).described_as("count").snapshot("baseline").is_equal_to(EXPECTED_COUNT)  # Noncompliant

    assert_that(value()).is_equal_to(value())
    assert_that(42).is_equal_to(EXPECTED_COUNT)
    assert_that(value()).is_equal_to()
    assert_that().is_equal_to(value())
    assert_that(value()).is_not_equal_to(42)
    builder.is_equal_to(42)
    factory().is_equal_to(42)
    other_call(value()).is_equal_to(42)
