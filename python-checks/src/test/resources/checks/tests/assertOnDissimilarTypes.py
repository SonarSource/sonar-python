import unittest

class A:
    pass

class B:
    pass

class C:
    pass

class D:
    pass

class ClassWithEq:
    def __eq__(self, other):
        pass

class ClassWithNe:
    def __ne__(self, other):
        pass

class MyTest(unittest.TestCase):
  def test_assert_is_non_compliant(self):
    myString = "toto"
    myInt = 3
    a = A()
    b = B()
    aInd = A()
#   ^^^^^^^^^^> {{Last assignment of "aInd"}}
    bInd = B()
#   ^^^^^^^^^^> {{Last assignment of "bInd"}}

    self.assertIs(aInd, bInd) # Noncompliant {{Change this assertion to not compare dissimilar types (A and B).}}
#                 ^^^^^^^^^^
    aInd = A()
    self.assertEqual(first="string", second=True) # Noncompliant
    self.assertIs(msg="my message", first="myString", second=a) # Noncompliant
    self.assertIs(a, myString) # Noncompliant
    self.assertIs(a, "string") # Noncompliant
    self.assertIs(a, myInt) # Noncompliant
    self.assertIs(a, 5) # Noncompliant
    self.assertIs(a, 42.42) # Noncompliant
    self.assertIs(a, 42j) # Noncompliant
    self.assertIs('foo'.partition('f'), "42") # Noncompliant

  def test_assert_is_compliant(self):
    a = A()
    b = a
    c = None

    self.assertIs() # OK
    self.assertIs(a, A()) # OK
    self.assertIs(a, a) # OK
    self.assertIs(a, b) # OK
    self.assertIs(a, None) # OK
    self.assertIs(None, None) # OK
    self.assertIs(c, None) # OK
    self.assertIs(a, c) # OK
    self.assertIs("my string", "my string") # OK
    self.assertIs(42, 42) # OK
    self.assertIs("my string", "another string") # OK
    self.assertIs(42, 0) # OK
    self.assertIs(msg="my message", first=a, second=a) # OK

  def test_assert_equal_non_compliant(self):
    ab = A()
    cd = C()
    if x == 42:
        ab = B()
        cd = D()
    myString = "toto"
    myInt = 3
    a = A()
    b = B()
    classWithEq = ClassWithEq()
    classWithNe = ClassWithNe()
    mydict = {"x": a}

    self.assertEqual(a, ab)
    self.assertEqual("string", True) # Noncompliant
    self.assertEqual(A(), []) # Noncompliant
    self.assertEqual(A(), ()) # Noncompliant
    self.assertEqual(A(), {}) # Noncompliant
    self.assertEqual({42:''}, {1}) # Noncompliant
    self.assertEqual(A(), B()) # Noncompliant
    self.assertEqual(a, b) # Noncompliant
    self.assertEqual(a, myString) # Noncompliant
    self.assertEqual(a, "string") # Noncompliant
    self.assertEqual(a, myInt) # Noncompliant
    self.assertEqual(a, 5) # Noncompliant
    self.assertEqual(a, 42.42) # Noncompliant
    self.assertEqual(a, 42j) # Noncompliant
    self.assertEqual('foo'.partition('f'), "42") # Noncompliant

    # Test with UnionType argument for error message without type indication
    self.assertEqual(5, ab) # Noncompliant {{Change this assertion to not compare dissimilar types.}}
    self.assertEqual(ab, 5) # Noncompliant {{Change this assertion to not compare dissimilar types.}}
    self.assertEqual(ab, cd) # Noncompliant {{Change this assertion to not compare dissimilar types.}}

    self.assertEqual(acos(1), "0") # OK

    tuple = "name", "passwd", 123, 456, "gecos", "dir", "shell"
    passwd = pwd.struct_passwd(tuple)
    self.assertEqual(passwd, "something")
    passwd_2 = pwd.struct_passwd(tuple)
    self.assertEqual(passwd, passwd_2) # OK
    self.assertEqual(passwd, pwd.getpwuid(1)) # OK


  def test_assert_equal_compliant(self):
    myString = "toto"
    myInt = 3
    a = A()
    b = B()
    aa = a
    classWithEq = ClassWithEq()
    classWithNe = ClassWithNe()
    mydict = {"x": a}

    self.assertEqual(a, classWithEq) # OK
    self.assertEqual(a, classWithNe) # OK
    self.assertEqual(a, a) # OK
    self.assertEqual(a, None) # OK
    self.assertEqual(None, a) # OK
    self.assertEqual(None, None) # OK
    self.assertEqual(d, None) # OK
    self.assertEqual(a, aa) # OK
    self.assertEqual(classWithEq, classWithEq) # OK
    self.assertEqual(classWithEq, myString) # OK
    self.assertEqual(classWithEq, "string") # OK
    self.assertEqual(classWithEq, myInt) # OK
    self.assertEqual(classWithEq, 5) # OK
    self.assertEqual(classWithEq, 42.42) # OK
    self.assertEqual(classWithEq, 42j) # OK
    self.assertEqual(myString, classWithEq) # OK
    self.assertEqual("string", classWithEq) # OK
    self.assertEqual(myInt, classWithEq) # OK
    self.assertEqual(5, classWithEq) # OK
    self.assertEqual(42.42, classWithEq) # OK
    self.assertEqual(42j, classWithEq) # OK
    self.assertEqual('foo'.partition('f'), classWithEq) # OK
    self.assertEqual(A(), None) # OK
    self.assertEqual(None, A()) # OK
    self.assertEqual(A(), A()) # OK
    self.assertEqual(set([1, 2]), frozenset([1, 2])) # OK
    self.assertEqual({}, collections.OrderedDict()) # OK

  def test_other(self):
    *a, = 1, 2 # test unpacking argument instead of regular argument
    b = 5
    b += 1 # test compound assignment
    self.assertTrue(5 == 3)
    self.assertEqual(5)
    self.assertEqual()
    self.assertEqual(*a, 5)
    self.assertEqual(5, *a)
    self.assertEqual("5", b)  # Noncompliant
    a.assertEqual("string", True)
