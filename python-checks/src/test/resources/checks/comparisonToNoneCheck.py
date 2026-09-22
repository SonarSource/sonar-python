from typing import Any, Tuple, Dict
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

def kwargs_any(*args: Any, **some_dict:Any):
    if args is not None: ...
    if some_dict is not None: ...

def kwargs(*some_args: Any, **kwargs: Any):
    if kwargs is not None: ...
    if some_args is not None: ...

def lxml_find():
    from lxml import etree
    root = etree.fromstring(b"<response><status>ok</status></response>")
    handle = root.find(".//handle")
    assert handle is None
    if handle is not None:
        print(handle.text)

def lxml_findtext():
    from lxml import etree
    root = etree.fromstring(b"<response><status>ok</status></response>")
    if root.findtext(".//handle") is None:
        print("handle is missing")
    tree = etree.parse("response.xml")
    if tree.findtext(".//handle") is None:
        print("handle is missing")

def lxml_tree_find():
    from lxml import etree
    tree = etree.parse("response.xml")
    handle = tree.find(".//handle")
    if handle is not None:
        print(handle.text)

def lxml_getroot():
    from lxml import etree
    tree = etree.parse("response.xml")
    if tree.getroot() is None:
        print("root is missing")
