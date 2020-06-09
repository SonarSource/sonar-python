
def f_strings():
    var = 42
    f"{var}" # Ok
    f"[var]" # Noncompliant {{Add replacement fields or use a normal string instead of an f-string.}}
   #^^^^^^^^
    F"[var]" # Noncompliant
    f'[var]' # Noncompliant
    f"""[var]""" # Noncompliant
    fr"[var]" # Noncompliant
    "" # Ok - not an f-string
    "{var}" # Ok - not an f-string

    (f"a" f"b" f"c")  # Noncompliant {{Add replacement fields or use a normal string instead of an f-string.}}
    #^^^^^^^^^^^^^^
    (f"{var}" f"b" f"c") # Ok

    (f"a" + f"b" + f"c") # FN
    (f"{var}" + f"b" + f"c") # Ok

def f_string_edge_case():
    2 + 2
    2 + f"{var}"
    f"{var}" + 2
    2 + f"" + f"" + 2
    f"" + (f"" + f"") # FM

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

import logging as renamed_logging
renamed_logging.error("Foo %s", "Bar", 'Too many')  # Noncompliant

module_logger = logging.getLogger('mylogger')
module_logger.error("Foo %s", "Bar")
module_logger.error("Foo %s", "Bar", "too many")  # FN

def fun():
    pass

def edge_case():
    fun = 'hello'
    fun('')
    no_such_function('hello')
