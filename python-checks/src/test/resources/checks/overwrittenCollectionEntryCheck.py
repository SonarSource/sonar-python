def numeric_key(list1, foo):
    list1[1] = "a"
    list1[2] = "a"
#        ^^^> {{Original value.}}
    list1[2] = "a" # Noncompliant {{Verify this is the key that was intended; a value has already been saved for it on line 3.}}
#        ^^^
    list1[3] = "a"
    list1[4] = "a"
    list1[1] = "a" # Noncompliant {{Verify this is the key that was intended; a value has already been saved for it on line 2.}}
    list1[6] = "a"
    foo(list1)
    list1[6] = "a"

def string_key(list1):
    list1["k1"] = "a"
    list1["k2"] = "a"
    list1["k1"] = "a" # Noncompliant

def variable_key(list1, index1, index2):
    list1[index1] = "a"
    list1[index2] = "a"
    index2 = 42
    list1[index2] = "a"
    list1[index2] = "a" # Noncompliant

def other_kinds_of_key(list1, foo):
    list1[foo()] = "a"
    list1[foo()] = "a"
    list1[foo().bar] = "a"
    list1[foo().bar] = "a"

def different_collections(list1, list2):
    list1[1] = "a"
    list2[1] = "a"
    list1[2] = "a"
    list2[1] = "a" # Noncompliant

def nested_subscription_expressions(list1, foo):
    list1[1][1] = "a"
    list1[1][2] = "a"
    list1[2][1] = "a"
    list1[2][1] = "a" # Noncompliant
#        ^^^^^^
    list1[1][foo.bar] = "a"
    list1[1][foo.qix] = "a"
    list1[foo.bar][1] = "a"
    list1[foo.qix][1] = "a"


def tuple_keys(list1):
    list1[1, 2] = "a"
    list1[1, 3] = "a"
    list1[1, 3] = "a" # Noncompliant
    list1[1, 4] = "a"
    list1[5] = "a"
    list1[5,] = "a"

def slicings(list1):
    list1[1:] = "a"
    list1[:1] = "a"
    list1[1:2] = "a"
    list1[1:3] = "a"
    list1[2:1] = "a"
    list1[1:3] = "a" # Noncompliant
    list1[1:2:3] = "a"
    list1[1:2:4] = "a"
    list1[1:2:3] = "a" # Noncompliant
    list1[:2:3] = "a"
    list1[:2:4] = "a"
    list1[:2:3] = "a" # Noncompliant

def negative_indices(list1, f):
    list1[-1] = "a"
    list1[-2] = "a"
    list1[-2] = "a" # Noncompliant
    list1[-5:-3] = "a"
    list1[-7:-5] = "a"
    list1[-7:-5] = "a" # Noncompliant
    list1[-9:-f()] = "a"
    list1[-9:-f()] = "a"

def non_trivial_collections(f, obj1, obj2):
    f()[1] = "a"
    f()[1] = "a"
    obj1.foo[1] = "a"
    obj2.foo[1] = "a"
    obj2.foo[1] = "a" # Noncompliant
    xxx1.foo[1] = "a"
    xxx2.foo[1] = "a"

def valid_replacement(list1, foo, x):
    list1["foo"] = foo(x)
    list1["foo"] = list1["foo"].replace("a", "b")

def used_collection(list1):
    list1[1] = 42
    list1[1] = foo(list1[1])

    list1[2] = 42
    list1[2] = foo(list1)

    list[3] = 42
    list[3] = list[1] # FN

def same_key_multiple_collections(bar_list, foo_entry, bar_entry, foo_value, bar_value):
    import pandas as pd
    a = {}
    a.foo_matrix = pd.DataFrame(index=bar_list, columns=bar_list)
    a.bar_matrix = pd.DataFrame(index=bar_list, columns=bar_list)

    a.foo_matrix.loc[foo_entry, bar_entry] = foo_value
    a.foo_matrix.loc[foo_entry, bar_entry] = foo_value # Noncompliant
    a.foo_matrix.at[foo_entry, bar_entry] = foo_value

class Record:
    def __init__(self, names):
        """Initialize class."""
        self._names = names
        self._count_info = {}
        self._label_info = {}
        for name in self._names:
            self._count_info[name] = 0
            self._count_info[name] = 2 # Noncompliant
            self._label_info[name] = None

def sub_collection():
    defs = anno.getanno(symbol_a, anno.Static.ORIG_DEFINITIONS)
    defs[0].directives[directive_key] = {
        'test_arg': parser.parse_expression('foo'),
        'other_arg': parser.parse_expression('bar'),
    }
    defs[1].directives[directive_key] = { # Noncompliant
        'test_arg': parser.parse_expression('foo'),
        'other_arg': parser.parse_expression('baz'),
    }

