class A:
    pass

def format_syntax():
    '%s %()s %(name)s' % ('one', 'two', 'three') # Noncompliant {{Use only positional or only named fields, don't mix them.}}
   #^^^^^^^^^^^^^^^^^^
    '%q' % 42 # Noncompliant {{Fix this formatted string's syntax.}}
    'Hello % ' % 'name' # Noncompliant {{Fix this formatted string's syntax.}}
   #^^^^^^^^^^
    "%" % "str"  # Noncompliant
    "%(key)" % {"key": "str"}  # Noncompliant
    "%(keys" % {"key": "str"}  # Noncompliant
    "%#3.5lo" % 42  # Ok
    "%?3.5lo" % 42  # Noncompliant
    "%#?.5lo" % 42  # Noncompliant
    "%#3?5lo" % 42  # Noncompliant
    "%#3.?lo" % 42  # Noncompliant
    "%#3.5?o" % 42  # Noncompliant
    "%#3.5l?" % 42  # Noncompliant

def tuple_arguments():
    '%s %s %s %s' % ('one', 'two') # Noncompliant {{Add 2 missing argument(s).}}
                   #^^^^^^^^^^^^^^
    '%s %s' % ('one', 'two', 'three') # Noncompliant {{Remove 1 unexpected argument(s).}}
             #^^^^^^^^^^^^^^^^^^^^^^^
    '%(first)s %(second)s' % ('one', 'two') # Noncompliant {{Replace this formatting argument with a mapping.}}
                            #^^^^^^^^^^^^^^
    '%%%s' % ('one') # Ok
    t = ('one', 'two')
    '%s %s %s' % t # FN

def converters():
    '%s' % (A(),) # Ok
    '%d' % (1,)  # Ok
    '%d' % (1.2,)  # Ok
    '%d' % (A(),)  # Noncompliant {{Replace this value with a number as "%d" requires.}}
           #^^^
    '%o' % (1,)  # Ok
    '%o' % (1.2,)  # Noncompliant {{Replace this value with an integer as "%o" requires.}}
           #^^^
    '%c' % ("a",)  # Ok
    '%c' % (1,)  # Ok
    '%c' % (1.5,)  # Noncompliant {{Replace this value with an integer or a single character string as "%c" requires.}}
    '%c' % ("ab",)  # Noncompliant
    '%c' % (A(),)  # Noncompliant

    x = 'c'
    x = 2.5
    '%c' % (x, ) # Noncompliant
    a_string = 'a string'
    '%d' % a_string  # Noncompliant


def width_precision():
    "%*.*le" % (3, 5, 1.1234567890)  # Ok
    "%*.*le" % (3.2, 5, 1.1234567890)  # Noncompliant {{Replace this value with an integer as "*" requires.}}
               #^^^
    "%*.*le" % (3, 5.2, 1.1234567890)  # Noncompliant
    "%*.*le %s %.*e" % (3, 5, 1.1234567890, "a string", 3.3, 0.987654321)  # Noncompliant


def dict_arguments():
    '%(first)s %(second)s %(third)s' % {'first': 'one', 'second': 'two'} # Noncompliant {{Provide a value for field "third".}}
    "%(a)s %(b)s %(c)s" % {"a": "str"}  # Noncompliant 2
    '%(first)s' % {'first': 'one', 'second': 'two'} # Ok - this is in the scope of S3457
    '%s %s' % {'first': 'one', 'second': 'two'} # Noncompliant {{Replace this formatting argument with a tuple.}}
             #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    '%(first)s' % {'first': A()} # Ok
    '%(first)d' % {'first': A()} # Noncompliant
                           #^^^
    '%(first)s %(first)s' % {'first': 'foo'} # Ok
    '%(first)s %(first)s %(second)s' % {'first': 'foo'} # Noncompliant
    '%(first)s %(first)s %(second)s' % {'first': 'foo', 'second': 'bar', 'third': 'baz'} # Ok (S3457)
    "%(key)*s" % {"key": "str"}  # Noncompliant

    a = 'first'
    t = {'a': 'b'}
    '%(first)s %(second)s' % {**t} # FN
    '%(first)d %(second)d' % {a: 'foo', 'second': 'bar'} # FN
    '%(first)s %(third)d' % {'first': 'foo', 'second': 'bar'} # Noncompliant {{Provide a value for field "third".}}
                           #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    '%(a)d %(a)s' % {"a": 1}  # Ok
    '%(a)d %(a)s' % {"a": "str"}  # Noncompliant
    "%%(key)s" % {"other": "str"} # Ok

def single_arguments():
    '%d' % 42 # Ok
    '%d' % '42' # Noncompliant
    '%d' % (42) # Ok
    '%d' % ('42') # Noncompliant

def other():
    f1 = "%d"
    f1 % ('hello') # Noncompliant
    f2 = "%#3?5lo"
        #^^^^^^^^^>
    f2 % ('hello') # Noncompliant {{Fix this formatted string's syntax.}}
   #^^
    "%(1)s" % {1: "str"} # Noncompliant {{Replace this key; %-format accepts only string keys.}}
              #^
    '%s %s' % ['a', 'b']  # Noncompliant {{Replace this formatting argument with a tuple.}}
    '%(field)s' % ['a'] # Noncompliant {{Replace this formatting argument with a mapping.}}
    '%(field)s' % 'a' # Noncompliant

    class Map:
        def __getitem__(self, item):
            pass

    '%(field)s' % Map() # Ok
    '%(foo)s %(bar)s' % A() # Noncompliant {{Replace this formatting argument with a mapping.}}

    ("%s" " %s" % (1,))  # Noncompliant
    ("%s" + " concatenated %d" % (1, 1))  # Noncompliant
    ("%s" + " concatenated" % (1,))  # Noncompliant {{Add replacement field(s) to this formatted string.}}
    ("%s" + " %s" % (1,))  # Ok
    args = ("are you", "a good day")
    "hello %s, how %s, this is %s" % ("friend", *args) # Ok
    args = ("are you")
    "hello %s, how %s, this is %s" % ("friend", *args) # FN

    field = '%s'
    f'{field} %s' % ('hello') # FN

def some_duck_typing():
    class MyCustomFloat:
        def __init__(self, val):
            self.val = val

        def __float__(self):
            return self.val

    # FP
    undercover_float = MyCustomFloat(42.3)
    "hello %f" % undercover_float # Noncompliant

def byte_formatting():
    # We do not handle byte formatting for now
    b'%s %s' % ('one') # FN
    b'(?P<%b>%b)' % (b'one') # FN

def edge_case():
    5 % 2
    x = 5
    x % ('hello')
    y = 5
    y = 6
    y % ('hello')
    '' % ('one') # Noncompliant {{Add replacement field(s) to this formatted string.}}
    '' % ([]) # Ok
    '' % [] # Ok

def unpacking_assignment():
  x, = 42,
  "%X" % x
