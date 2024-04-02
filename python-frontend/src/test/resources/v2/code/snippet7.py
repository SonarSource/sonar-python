# snippet 7
import time
class A:
    def method1(self) -> int:
        return 42

class B:
    def method1(self) -> str:
        return 42

class C(B):
    def method2(self) -> str:
        return 42

class D:
    def method2(self) -> str:
        return 42

if time.time() % 2 == 0:
    a = C()
else:
    a = A()

b = a

