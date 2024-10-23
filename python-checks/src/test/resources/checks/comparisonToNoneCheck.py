import pydoc
def identity_check(param):
    a = None
    b = 42
    if a is param: pass
    if a is None: pass # Noncompliant {{Remove this identity check; it will always be True.}}
    if a is not None: pass # Noncompliant {{Remove this identity check; it will always be False.}}
    if b is None: pass # Noncompliant {{Remove this identity check; it will always be False.}}
    if b is not None: pass # Noncompliant {{Remove this identity check; it will always be True.}}
    if a is b: pass # Noncompliant {{Remove this identity check; it will always be False.}}
    c = None
    if a is c: pass # Noncompliant {{Remove this identity check; it will always be True.}}
    d = "abc"
    if b is d: pass
    if None is a: pass # Noncompliant
    if None is b: pass # Noncompliant
    obj = pydoc.locate("test")
    if obj is not None: pass    
    if None is not obj: pass

    obj = object()
    if obj is not None: pass # FN
    if None is obj: pass # FN

    import xml.etree.ElementTree as ET
    tree = ET.ElementTree()
    if tree.getroot() is None: pass # Ok as getroot can return None

def equality_check(param):
    a = None
    b = 42
    if a == param: pass
    if a >  None: pass
    if a >= None: pass
    if a == None: pass # Noncompliant {{Remove this == comparison; it will always be True.}}
    if a != None: pass # Noncompliant {{Remove this != comparison; it will always be False.}}
    if b == None: pass # Noncompliant {{Remove this == comparison; it will always be False.}}
    if b != None: pass # Noncompliant {{Remove this != comparison; it will always be True.}}
    if a == b: pass # Noncompliant {{Remove this == comparison; it will always be False.}}
    c = None
    if a == c: pass # Noncompliant {{Remove this == comparison; it will always be True.}}
    d = "abc"
    if b == d: pass
    if b != d: pass
    obj = pydoc.locate("test")
    if obj == None: pass
    if None == obj: pass
    
    obj = object()
    if obj == None: pass # FN
    if None == obj: pass # FN

def both_can_be_none(cond1, cond2):
    if cond1:
        a = 42
    else:
        a = None

    if cond2:
        b = True
    else:
        b = None

    if a == b:
        pass

    if a is b:
        pass

def type_hints_are_skiped(a: int):
    if a is None: pass
    if None is a: pass
    if a == None: pass
    if None == a: pass

def next_in_loop(list_param):
    while True:
        p = next(list_param)
        if p is None: # Compliant
            break

def assert_is_none():
    import re
    _re_prop_name = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")
    assert _re_prop_name.match("_test") is None # Compliant

def no_issue_on_declared_type():
    class SomeClass:
        test = 3

    def test_method(a0: SomeClass) -> None:
        if a0 is None:
            return
        if a0.test is None:
            return

def getattr(self):
    func = getattr(self, "eventReceived_", None)
    if func == None: # Compliant
        pass

# TODO: SONARPY-2236 because @property are not handled, the following code will raise FPs
def class_property(arr):
    class A:
        @property
        def always_none(self):
            return None

        @property
        def never_none(self):
            return True

        @property
        def maybe_none(self, a):
            if a:
                return True
            return None
    a = A()

    if a.always_none is None:
        pass
    if a.never_none is None:
        pass
    if a.maybe_none is None:
        pass

    if None is a.always_none:
        pass
    if None is a.never_none:
        pass
    if None is a.maybe_none:
        pass

    if a.always_none == None:
        pass
    if a.never_none == None:
        pass
    if a.maybe_none == None:
        pass
    if None == a.always_none:
        pass
    if None == a.never_none:
        pass
    if None == a.maybe_none:
        pass

    v = memoryview(arr)
    if v.shape is None:
        pass

    @unittest.skipIf(arr is None)
    def some_func():
        return None

def reference_to_unkown_variable():
    if input is None:
        pass

    if some_unkown_variable == None:
        pass

def checks_with_classes(arr):
    from not_found_module import NotFoundClass
    class A(): pass
    if A() is None: #Noncompliant
        pass
    if A() == None: #Noncompliant
        pass

    if memoryview(arr) is None: #Noncompliant
        pass

    if memoryview(arr) == None: #Noncompliant
        pass

    if NotFoundClass() is None:
        pass

    @some_decorator()
    class DecoratedClass:
        pass

    decorated_class = DecoratedClass()
    if decorated_class is None: #Noncompliant
        pass

    if decorated_class == None: #Noncompliant
        pass

def use_after_class_declaration():
    class A: pass
    def method():
        a = A()
        if a is None: #Noncompliant
            pass

def if_in_class():
    class A: pass
    class B:
        if A() is None: #Noncompliant
            some_field = "test"
        if A() == None: #Noncompliant
            some_other_field = "test"
