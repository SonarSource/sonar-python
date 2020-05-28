class A:
    pass

def format_syntax():
    '%s %()s %(name)s' % ('one', 'two', 'three') # Noncompliant {{Use only positional or only named field, don't mix them.}}
   #^^^^^^^^^^^^^^^^^^
   '%q' % 42 # Noncompliant {{Fix this formatted string's syntax; %q is not a valid conversion type.}}


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
    '%c' % (x, ) # Ok - we do not know the type of 'x'


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

    a = 'first'
    t = {'a': 'b'}
    '%(first)s %(second)s' % {**t} # FN
    '%(first)d %(second)d' % {a: 'foo', 'second': 'bar'} # FN
    '%(first)s %(third)d' % {'first': 'foo', 'second': 'bar'} # Noncompliant {{Provide a value for field "third".}}
                           #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def single_arguments():
    '%d' % 42 # Ok
    '%d' % '42' # Noncompliant
    '%d' % (42) # Ok
    '%d' % ('42') # Noncompliant


def edge_case():
    5 % 2
