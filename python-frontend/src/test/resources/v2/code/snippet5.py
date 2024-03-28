# snippet 5
class A:
    def method1(self) -> int:
        int_value = 10
        return int_value

class B(A):
    def method2(self) -> str:
        str_value = ""
        return str_value

class C:
    def method3(self):
        a = ""
        return a

a_instance = B()
