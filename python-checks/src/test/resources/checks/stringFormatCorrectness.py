# Due to parsing errors with formatted expressions, the rule is disabled for f-strings: see SONARPY-726
def f_strings():
    var = 42
    f"{var}" # Ok
    f"[var]" # FN {{Add replacement fields or use a normal string instead of an f-string.}}

def printf_style():
    "%(key)s" % {"key": "str", "other": "key"}  # Noncompliant {{Remove this unused argument or add a replacement field.}}
                              #^^^^^^^
    "%%(key)s" % {"key": "str"}  # Noncompliant
    d = {'a': 'foo', 'b': 'bar'}
    "%(key)s" % d # FN
    "%(key)s" % {'key': 'a', **d} # FN
    "%(key)s" % {d: 'value'}


    "%?3.5lo" % ('one') # Ok, this is in the scope of S2275

def str_format_style():
    '{} {}'.format('one', 'two', 'three')  # Noncompliant {{Remove this unused argument.}}
                                #^^^^^^^
    '{:{}} {}'.format('one', 's', 'three', 'four')  # Noncompliant
    '{0:{1}} {2}'.format('one', 's', 'three', 'four')  # Noncompliant
                                             #^^^^^^
    '{0[{}]}'.format({"{}": 0})  # Ok
    '{a}'.format(a=1, b=2)  # Noncompliant
    '{a:{b}} {c}'.format(a='one', b='s', c='three', d='four')  # Noncompliant
                                                   #^^^^^^^^
    ("{}" " concatenated".format(1))  # Ok
    ("{}" + " concatenated".format(1))  # Noncompliant

    dict_data = {'one': '1', 'two': 2}
    tuple_data = ("1", 2, 3)
    '{}'.format(*tuple_data)  # FN
    '{one}'.format(**dict_data)  # FN

    '{0'.format('one') # OK, this is in the scope of S2275

import logging

def logger_format():
    logging.error("%?3.5lo blabla", 42) # Noncompliant {{Fix this formatted string's syntax.}}
                 #^^^^^^^^^^^^^^^^
    logging.error("%(key)s %s", {"key": "str", "other": "str"})  # Noncompliant {{Use only positional or only named fields, don't mix them.}}
                 #^^^^^^^^^^^^
    logging.error('%(a)s %(b)s', ['a', 'b'], 42)  # Noncompliant {{Replace formatting argument(s) with a mapping; Replacement fields are named.}}
                                #^^^^^^^^^^^^^^
    logging.error('%d', '42')  # Noncompliant {{Replace this value with a number as "%d" requires.}}
                       #^^^^
    logging.error('%(key)d', {'key': '42'}) # Noncompliant {{Replace this value with a number as "%d" requires.}}
                                    #^^^^
    logging.error("%(a)s %(b)s", {"a": 1, "b": 2}, 3)  # Noncompliant {{Change formatting arguments; the formatted string expects a single mapping.}}

    logging.error("%(a)s %(b)s", {"a": 1}) # Noncompliant {{Provide a value for field "b".}}
                                #^^^^^^^^
    logging.error("%s") # Noncompliant {{Add argument(s) corresponding to the message's replacement field(s).}}
                 #^^^^

    logging.error("%(a)s", {"a": 1, "b": 2}) # OK - this will not log an error
    logging.error("%d", 42, exc_info=True) # Ok

    logging.error("") # Ok
    logging.error("", 'too', 'many') # Noncompliant

    d = {'key': '42'}
    logging.error('%(key)d', d) # FN

    f1 = "%?3.5lo blabla"
        #^^^^^^^^^^^^^^^^>
    logging.error(f1, 42) # Noncompliant {{Fix this formatted string's syntax.}}
                 #^^

    f2 = "%s"
    f2 = "%d"
    logging.error(f2, '42') # FN

    logging.error()
    t = ('one', 'two')
    logging.error("%d", *t) # FN
    logging.error(msg="%d", kw1='42')

    logging.error("This is valid % % %") # Ok

import logging as renamed_logging
renamed_logging.error("Foo %s", "Bar", 'Too many')  # Noncompliant

l1 = logging.getLogger('l1')
l1.error("Foo %s", "Bar")
l1.error("Foo %s", "Bar", "too many")  # Noncompliant

l2 = logging.getLogger('l2')
l2 = logging.getLogger('l3')
l2.error("Foo %s", "Bar")
l2.error("Foo %s", "Bar", "too many")  # FN

l3 = 'hello'
l3.error("Foo %s", "Bar", "too many")

x = [l1, l2]
x[0].error("Foo %s", "Bar", "too many")  # FN

def logger_in_function():
    local_logger = logging.getLogger('mylogger')
    local_logger.error("Foo %s", "Bar")
    local_logger.error("Foo %s", "Bar", "too many")  # Noncompliant

def fun():
    pass

def edge_case():
    fun = 'hello'
    fun('')
    no_such_function('hello')
    l4 = no_such_function('')
    l4.error("Foo %s", "Bar", "too many")
