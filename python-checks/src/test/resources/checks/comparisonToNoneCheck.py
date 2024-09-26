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

