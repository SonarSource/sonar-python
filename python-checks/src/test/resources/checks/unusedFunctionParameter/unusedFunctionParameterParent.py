class Parent:
    # Noncompliant@+2 {{Remove the unused function parameter "a".}}
    'Issues are not raised when the variable is mentioned in a comment related to the function'
    def do_something(self, a, b):
        #                  ^
        return compute(b)

    def do_something_else(self, a, b):
        return compute(a + b)

    def using_child_method(self):
        return self.method_defined_in_child_class_only(1)
