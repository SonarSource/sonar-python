import unittest

class ConstantTrueFalseTests(unittest.TestCase):
    """Rule S5914 should raise issues on cases where S5797 raises."""

    def test_constant_assert_true_with_literal(self):
        """assertTrue and assertFalse called on literals will always succeed or always fail."""
        self.assertTrue(True)  # Noncompliant {{Replace this expression; its boolean value is constant.}}
    #                   ^^^^
        self.assertTrue(False)  # Noncompliant
        self.assertTrue(42)  # Noncompliant
        self.assertTrue('a string')  # Noncompliant
        self.assertTrue(b'bytes')  # Noncompliant
        self.assertTrue(42.0)  # Noncompliant
        self.assertTrue({})  # Noncompliant
        self.assertTrue({"a": 1, "b": 2})  # Noncompliant
        self.assertTrue({41, 42, 43})  # Noncompliant
        self.assertTrue([])  # Noncompliant
        self.assertTrue([41, 42, 43])  # Noncompliant
        self.assertTrue((41, 42, 43))  # Noncompliant
        self.assertTrue(())  # Noncompliant
        self.assertTrue(None)  # Noncompliant

        # Same for assertFalse
        self.assertFalse(True)  # Noncompliant


    def test_constant_assert_true_with_literal(self):
        immutable_value = True
        foo(immutable_value)
        self.assertTrue(immutable_value)  # Noncompliant

        list_object = []
        foo(list_object)
        self.assertTrue(list_object)  # Compliant because the object can be changed




    def test_assert_statement(self):
        """The assert statement should be analyzed the same way as unittest.assertTrue with one exception.

        When the assert statement is used directly on a two elements literal tuple the issue should be raised
        only by RSPEC-5905 "Assert should not be called on a tuple literal". This is a very common mistake
        and having a dedicated rule will make it easier for developer to understand the issue.
        """
        assert True # Noncompliant {{Replace this expression; its boolean value is constant.}}
    #          ^^^^

        assert (1, "message")  # Ok. Issue raised by RSPEC-5905

        assert 1  # Noncompliant
        assert "foo"  # Noncompliant

    def test_assert_statement_used_for_test_failure(self):
        """Assert False or Assert 0 is often used to make a test fail.
        Usually it is better to use another assertion or throw an AssertionException.
        However, this rule is not intended to check this best practice."""
        # Should be raised by another rule
        assert False
        assert 0


    def test_constant_assert_true_with_unpacking(self):
        """Rule handles unpacking as S5797."""
        li = [1,2,3]
        di = {1:1}
        self.assertTrue([*li])  # False Negative. We don't check the size of unpacked iterables
        self.assertTrue([1, *li])  # Noncompliant

        self.assertTrue((*li))  # False Negative. We don't check the size of unpacked iterables
        self.assertTrue((1, *li))  # Noncompliant

        self.assertTrue({*li})  # False Negative. We don't check the size of unpacked iterables
        self.assertTrue({1, *li})  # Noncompliant

        self.assertTrue({**di})  # False Negative. We don't check the size of unpacked iterables
        self.assertTrue({2:3, **di})  # Noncompliant


    def test_constant_assert_true_with_module_and_functions(self):
        self.assertTrue(round)  # False Negative.
        self.assertTrue(unittest)  # False Negative.


    def test_constant_assert_true_with_class_and_methods_and_properties(self):
        class MyClass:
            def mymethod(self):
                if self.mymethod:
                    pass

            @property
            def myprop(self):
                pass

        myinstance = MyClass()
        self.assertTrue(MyClass)  # Noncompliant
        self.assertTrue(MyClass.mymethod)   # Noncompliant
        self.assertTrue(myinstance.mymethod)   # Noncompliant
        self.assertTrue(myinstance.myprop)   # Ok
        self.assertTrue(myinstance)   # Ok

    def test_constant_assert_true_with_generators_and_lambdas(self):
        lamb = lambda: None
        self.assertTrue(lamb)  # Noncompliant

        gen_exp = (i for i in range(42))
        self.assertTrue(gen_exp)  # Noncompliant

        def generator_function():
            yield
        generator = generator_function()
        self.assertTrue(generator)  # False Negative. Not covered by S5797 for now.

    def test_constant_assert_true_with_variables_pointing_to_single_immutable_type_value(self):
        """For immutable types we consider that if a variable can only have one value and it is used as a condition, we should raise an issue.
        See S5797 for more info."""
        if 42:
            an_int = 1
        else:
            an_int = 2

        an_int = 0  # Overwrite all previus values
        self.assertTrue(an_int)  # Noncompliant

    def test_empty_assertion(self):
        self.assertTrue()

#
# RSPEC_5727: Comparison to None should not be constant
#

def a_function() -> int:
    return None

class ConstantNoneTests(unittest.TestCase):
    """Rule S5914 should raise issues on cases where S5727 raises."""
    def test_constant_assert_none(self):
        myNone = None
        self.assertIsNone(myNone)  # Noncompliant {{Remove this identity assertion; its value is constant.}}
#       ^^^^^^^^^^^^^^^^^^^^^^^^^
        self.assertIsNotNone(myNone)  # Noncompliant

    def test_ignore_annotations(self):
        """This rule should ignore type annotations when they are not comming from Typeshed. It
        is ok to test if a function behaves as expected."""
        self.assertIsNotNone(a_function())  # Ok. Avoid False Positive here.


#
# RSPEC_5796: New objects should not be created only to check their identity
#

class ConstantNewObjectTests(unittest.TestCase):
    """Rule S5914 should raise issues on cases where S5796 raises."""
    def helper_constant_assert_new_objects(self, param):
        self.assertIs(param, [1, 2, 3])  # Noncompliant
        self.assertIsNot(param, [1, 2, 3])  # Noncompliant
        self.assertIsNot([1, 2, 3], param)  # Noncompliant

# SONARPY-1102 : Fix FP on S5914 for nonlocal variables
# The "nonlocal" keyword define that the variable we want to use is the same one as defined in upper scope.
# In case of nonlocal variable assignment, our current Dataflow Analysis does not allow us to detect a changing variable
# Chosen solution is to ignore variable for which there is a nonlocal assignment somewhere
class NonLocalTest(unittest.TestCase):
    def test_nonlocal_variables(self):
        class NonLocalCallClass:
            def close(self):
                nonlocal socket_closed
                socket_closed = True
        socket_closed = False
        x = SomeClass()
        x.close()
        self.assertTrue(socket_closed)

    def test_nonlocal_variables_unused(self):
        class NonLocalCallClass:
            def close(self):
                nonlocal socket_closed
                socket_closed = True
        socket_closed = False
        # Accepted FN : we don't detect that the nonlocal assignment is never called
        self.assertTrue(socket_closed)

# Covering edge case
class EdgeCase(unittest.TestCase):
    def test_nonlocal_variables(self):
        self.assertTrue(undefined_variable)
