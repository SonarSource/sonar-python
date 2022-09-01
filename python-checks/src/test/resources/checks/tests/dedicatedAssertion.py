import unittest

class SomeTest(unittest.TestCase):
    def assert_true_on_comparisons(self):
        a = 42
        b = foo()
        self.assertTrue(a == b)  # Noncompliant {{Consider using "assertEqual" instead.}}
        self.assertTrue(msg="fail", expr=a == b)  # Noncompliant {{Consider using "assertEqual" instead.}}
        self.assertTrue((a == b))  # Noncompliant {{Consider using "assertEqual" instead.}}
        self.assertTrue(a != b)  # Noncompliant {{Consider using "assertNotEqual" instead.}}
        self.assertTrue(a > b)  # Noncompliant {{Consider using "assertGreater" instead.}}
        self.assertTrue(a < b)  # Noncompliant {{Consider using "assertLess" instead.}}
        self.assertTrue(a >= b)  # Noncompliant {{Consider using "assertGreaterEqual" instead.}}
        self.assertTrue(a <= b)  # Noncompliant {{Consider using "assertLessEqual" instead.}}

    def no_fp_on_comparison_chaining(self):
        self.assertTrue(a < b < c)  # OK, is equivalent to self.assertTrue(a < b and b < c)

    def assert_true_on_function_call(self):
        self.assertTrue(isinstance(a,b)) # Noncompliant {{Consider using "assertIsInstance" instead.}}

    def assert_false_on_comparisons(self):
        a = 42
        b = foo()
        self.assertFalse(a == b)  # Noncompliant {{Consider using "assertNotEqual" instead.}}
        self.assertFalse(a != b)  # Noncompliant {{Consider using "assertEqual" instead.}}

    def assert_false_on_comparisons(self):
        self.assertFalse(isinstance(a, b))  # Noncompliant {{Consider using "assertNotIsInstance" instead.}}

    def assert_true_on_is_expression(self):
        self.assertTrue(a is b)  # Noncompliant {{Consider using "assertIs" instead.}}
        self.assertTrue(a is not b)  # Noncompliant {{Consider using "assertIsNot" instead.}}

    def assert_true_or_false_on_in_expression(self):
        self.assertTrue(a in b)  # Noncompliant {{Consider using "assertIn" instead.}}
        self.assertTrue(a not in b) # Noncompliant {{Consider using "assertNotIn" instead.}}
        self.assertFalse(a in b)  # Noncompliant {{Consider using "assertNotIn" instead.}}
        self.assertFalse(a not in b)  # Noncompliant {{Consider using "assertIn" instead.}}

    def assert_equal_on_boolean_literal(self):
        self.assertEqual(a, True)  # Noncompliant {{Consider using "assertTrue" instead.}}
        self.assertEqual((a), True)  # Noncompliant {{Consider using "assertTrue" instead.}}
        self.assertEqual(a, (True))  # Noncompliant {{Consider using "assertTrue" instead.}}
        self.assertEqual(True, a)  # Noncompliant {{Consider using "assertTrue" instead.}}
        self.assertEqual(True, True)  # Noncompliant {{Consider using "assertTrue" instead.}}
        self.assertEqual(a, False)  # Noncompliant {{Consider using "assertFalse" instead.}}
        self.assertEqual(False, a)  # Noncompliant {{Consider using "assertFalse" instead.}}
        self.assertEqual(False, False)  # Noncompliant {{Consider using "assertFalse" instead.}}
        self.assertEqual(True, False)  # Noncompliant {{Consider using "assertTrue" instead.}}
        self.assertEqual(False, True)  # Noncompliant {{Consider using "assertFalse" instead.}}
        my_true = True
        self.assertEqual(a, my_true)  # OK
        self.assertEqual(my_true, a)  # OK

    def assert_equal_on_none(self):
        self.assertEqual(x, None)  # Noncompliant {{Consider using "assertIsNone" instead.}}
        self.assertEqual(None, x)  # Noncompliant {{Consider using "assertIsNone" instead.}}

    def assert_equal_on_round(self):
        self.assertEqual(x, round(y, z))  # Noncompliant {{Consider using "assertAlmostEqual" instead.}}
        self.assertEqual(round(y, z), x)  # Noncompliant {{Consider using "assertAlmostEqual" instead.}}

    def assert_not_equal(self):
        self.assertNotEqual(x, None)  # Noncompliant {{Consider using "assertIsNotNone" instead.}}
        self.assertNotEqual(None, x)  # Noncompliant {{Consider using "assertIsNotNone" instead.}}
        self.assertNotEqual(x, round(y, z))  # Noncompliant {{Consider using "assertNotAlmostEqual" instead.}}
        self.assertNotEqual(x, True)
        self.assertNotEqual(True, x)
        self.assertNotEqual(False, x)

    def assert_almost_equal(self):
        self.assertAlmostEqual(a,b,c)  # OK
        self.assertAlmostEqual(a,round(b, c))  # Noncompliant {{Consider using the "places" argument of "assertAlmostEqual" instead.}}
        self.assertAlmostEqual(a,(round(b, c)))  # Noncompliant {{Consider using the "places" argument of "assertAlmostEqual" instead.}}
        self.assertAlmostEqual(round(b, c), a)  # Noncompliant {{Consider using the "places" argument of "assertAlmostEqual" instead.}}
        self.assertNotAlmostEqual(a,round(b, c))  # Noncompliant {{Consider using the "places" argument of "assertNotAlmostEqual" instead.}}
        self.assertNotAlmostEqual(round(b, c), a)  # Noncompliant {{Consider using the "places" argument of "assertNotAlmostEqual" instead.}}
        self.assertAlmostEqual(msg="fail", first=x, second=round(y,z))  # Noncompliant
        self.assertAlmostEqual(msg="fail", one=x, second=round(y,z))  # OK
        self.assertAlmostEqual(msg="fail", first=x, two=round(y,z))  # OK
        self.assertAlmostEqual()  # OK
        self.assertAlmostEqual(unknown.round(b, c), a)  # OK
        self.assertAlmostEqual(one=round(y,z), two=x)  # OK
        self.assertAlmostEqual(msg=round(), first=1, second=2)  # OK

    def invalid_assert(self):
        x.y.z.assertTrue(a == b)
        other.assertTrue(a == b)  # OK
        self.assertTrue
        self.assertTrue()
        self.assertTrue(msg=a==b, expr="yes")
        self.assertCountEqual(a, b)
        self.assertTrue(**a)
        self.assertTrue(a << b)
        self.assertTrue(a <> b)
        self.assertTrue(foo.bar())
        self.assertTrue(foo())
        self.assertEqual()
        self.assertEqual(a)
        self.assertEqual(**a, b)
        self.assertEqual(a, **b)


class AnotherClass:
    def not_a_test(self):
        self.assertTrue(a == b)  # OK
