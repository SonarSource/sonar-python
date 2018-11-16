class MyClass:
    def correct_method_name():
        pass

    def Incorrect_Method_Name():  # Noncompliant {{Rename method "Incorrect_Method_Name" to match the regular expression ^[a-z_][a-z0-9_]{2,}$.}}
#       ^^^^^^^^^^^^^^^^^^^^^
        pass

    def long_method_name_is_still_correct():
        pass

class MyTestCase(unittest.TestCase):
    def setUp(self): # ok, potentially overridden method, the name can't be changed
        self.message = 'hello'

def This_Is_A_Function():
    pass
