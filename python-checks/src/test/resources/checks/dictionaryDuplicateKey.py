def strings():
  {"one": 1, "two": 2}
  {"one": 1, "two": 2, "one": 3}  # Noncompliant {{Change or remove duplicates of this key.}}
  #^^^^^               ^^^^^<
  {"one": 1, "two": 2, 'one': 3}  # Noncompliant
  {"""multi
  line""": 1, # Noncompliant@-1
  "two": 2,
  """multi
  line""": 3}

  # We only raise if the strings have same values & prefixes, even if they would eventually evaluate to the same value
  {"one": 1, "two": 2, u"one": 3}  # FN (ok)
  {"one": 1, "two": 2, r"one": 3}  # FN (ok)
  {"on" "e": 1, "two": 2, "o" "ne": 2} # Noncompliant
  {"on" r"e": 1, "two": 2, "on" r"e": 2} # Noncompliant
  {"on" r"e": 1, "two": 2, "o" r"ne": 2} # FN (ok)

  # No issue on f-strings to avoid any risk of FP
  p = 1
  {f"one{p}": 1, "two": 2, f"one{p}": 3} # FN (ok)
  {f"one{p()}": 1, "two": 2, f"one{p()}": 3} # FN (ok)

def numbers():
  {1: "one", 2: "two", 3: "three"}
  {1: "one", 2: "two", 1: "three"}  # Noncompliant
  {1.0: "one", 2.0: "two", 3.0: "three"}
  {1.0: "one", 2.0: "two", 1.0: "three"}  # Noncompliant
  {0o1: "one", 0o2: "two", 0O1: "three"}  # Noncompliant
  {0x1: "one", 0x3: "two", 0X1: "three"}  # Noncompliant
  {0xB1E70073L: "1", 0xB1E70073L: "1"} # Noncompliant [[only valid for python 2]]
  {0xB1E70073l: "1", 0xB1E70073l: "1"} # Noncompliant [[only valid for python 2]]
  {0b1: "one", 0o2: "two", 0B1: "three"}  # Noncompliant
  {True: "one", False: "two", True: "three"}  # Noncompliant

def mixed_types():
  {1.0: "one", 2.0: "two", 0o1: "three"}  # Noncompliant
  {1: "one", 2: "two", 1.0: "three"}  # Noncompliant
  # True == 1
  {1: "one", 2: "two", True: "three"}  # Noncompliant
  # False == 0
  {0: "one", 2: "two", False: "three"}  # Noncompliant
  {"1": 1, 1: "1", "2": 2} # OK


def unpacking_expressions():
  dict = {1: 2}
  {1: 1, **dict} # FN
  {1: 1, **{1: 2}} # FN


def other_cases():
  {None: 1, None: 2}  # Noncompliant
  {(1, "2"): "one", 2: "two", (1, "2"): "three"}  # Noncompliant
  {(1, "2"): "one", 2: "two", ("2", 1): "three"}
  {1: {"a": 1}, 2: {"a": 2}}
  {1: {1: "a"}}


def repeated_variables(a1, a2, a3):
    {a1: 1, a2: 2, a1: 3}  # Noncompliant
    {a1.b: 1, a2: 2, a1.b: 3}  # Noncompliant
    {func: 1, a2: 2, func: 3}  # Noncompliant

    [{a: 1, a:2} for a in range(10)]  # Noncompliant

    class MyClass:
        pass

    {MyClass: 1, a2: 2, MyClass: 3}  # Noncompliant
    {MyClass.__doc__: 1, a2: 2, MyClass.__doc__: 3}  # Noncompliant

    {MyClass(): 1, a2: 2, MyClass(): 3}  # OK
    {a1(): 1, a2: 2, a1(): 3}  # OK
    {func(1, 2, 3): 1, a2: 2, func(1, 2, 3): 3}  # OK

def tuples():
  {(2, bar()): 1, (2, bar()): 2} # OK, function calls
  { (0o10, 'a'): 1, (8, "a"): 2 } # Noncompliant
  { (0o10, 'a'): 1, (8, "a", "b"): 2 } # OK
  { (1,): 1, 1 : 2 } # OK


def large_dict_literal():
  # Accepted FNs to avoid performance issues
  {
  "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b",
  "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b",
  "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b",
  "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b",
  "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b",
  "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b",
  "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b",
  "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b",
  "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b",
  "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b",
  "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b", "a": "b",
  "a": "b"
  }
