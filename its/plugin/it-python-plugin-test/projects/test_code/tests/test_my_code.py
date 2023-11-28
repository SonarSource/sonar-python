def test_something():
    assert 1 == 2
    assert (a, b)  # Issue here (S5905)
    if True:  # No issue here (S3923 doesn't run on test files)
      print("hello")
    else:
      print("hello")


from tests.my_other_test_file import MyTestClass

class SomeCustomTestCase(MyTestClass):
        def test_addition(self):
            self.assertTrue(1 + 2 == 3)  # Issue here (S5906)


from yest_another_test_file import MyOtherTestClass

class SomeOtherCustomTestCase(MyTestClass):
        def test_addition_2(self):
            self.assertTrue(1 + 2 == 3)  # Issue here (S5906)
