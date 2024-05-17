class Person:
    name: str
    age: int

    def foo(self):
        dynamically_assigned_variable = 42
        self.my_attribute = 24
        return "Hello World"


def instance_check():
    person = Person()
    person.foo()
