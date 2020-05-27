class A:
    pass

def tuple_arguments():
    '%s %s %s %s' % ('one', 'two') # Noncompliant {{Add 2 missing argument(s).}}
                   #^^^^^^^^^^^^^^
    '%s %s' % ('one', 'two', 'three') # Noncompliant {{Remove 1 unexpected argument(s).}}
             #^^^^^^^^^^^^^^^^^^^^^^^
    '%(first)s %(second)s' % ('one', 'two') # Noncompliant {{Replace this formatting argument with a mapping.}}
                            #^^^^^^^^^^^^^^

    t = ('one', 'two')
    '%s %s %s' % t # FN

def tuple_converters():
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

def single_arguments():
    '%d' % 42 # Ok
    '%d' % '42' # Noncompliant
