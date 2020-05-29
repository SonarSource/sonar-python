class A:
    pass

def format_syntax():
    '%s %()s %(name)s' % ('one', 'two', 'three') # Noncompliant {{Use only positional or only named field, don't mix them.}}
   #^^^^^^^^^^^^^^^^^^
    '%q' % 42 # Noncompliant {{Fix this formatted string's syntax; %q is not a valid conversion type.}}
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
    '' % ('one') # FN

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
    '%c' % (x, ) # Ok - we do not know the type of 'x'


def width_precision():
    "%*.*le" % (3, 5, 1.1234567890)  # Ok
    "%*.*le" % (3.2, 5, 1.1234567890)  # Noncompliant
    "%*.*le" % (3, 5.2, 1.1234567890)  # Noncompliant
    "%*.*le %s %.*e" % (3, 5, 1.1234567890, "a string", 3.3, 0.987654321)  # Noncompliant


def dict_arguments():
    '%(first)s %(second)s %(third)s' % {'first': 'one', 'second': 'two'} # Noncompliant {{Add 1 missing argument(s).}}
                                      #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    '%(first)s' % {'first': 'one', 'second': 'two'} # Noncompliant {{Remove 1 unexpected argument(s).}}
                 #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    '%s %s' % {'first': 'one', 'second': 'two'} # Noncompliant {{Replace this formatting argument with a tuple.}}
             #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    '%(first)s' % {'first': A()} # Ok
    '%(first)d' % {'first': A()} # Noncompliant
                           #^^^
    '%(first)s %(first)s' % {'first': 'foo'} # Ok
    '%(first)s %(first)s %(second)s' % {'first': 'foo'} # Noncompliant
    '%(first)s %(first)s %(second)s' % {'first': 'foo', 'second': 'bar', 'third': 'baz'} # Noncompliant
    "%(key)*s" % {"key": "str"}  # Noncompliant

    a = 'first'
    t = {'a': 'b'}
    '%(first)s %(second)s' % {**t} # FN
    '%(first)d %(second)d' % {a: 'foo', 'second': 'bar'} # FN
    '%(first)s %(third)d' % {'first': 'foo', 'second': 'bar'} # Noncompliant {{Provide a value for field "third".}}
                           #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    '%(a)d %(a)s' % {"a": 1}  # Ok
    '%(a)d %(a)s' % {"a": "str"}  # Noncompliant

def single_arguments():
    '%d' % 42 # Ok
    '%d' % '42' # Noncompliant
    '%d' % (42) # Ok
    '%d' % ('42') # Noncompliant


def edge_case():
    5 % 2
