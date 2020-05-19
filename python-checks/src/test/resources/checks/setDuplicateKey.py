def strings():
  {"one", "two"}
  {"one", "two", "one"}  # Noncompliant {{Change or remove duplicates of this key.}}
  #^^^^^         ^^^^^<
  {"one", "two", 'one'}  # Noncompliant
  {"""multi
  line""", # Noncompliant@-1
  "two",
  """multi
  line"""}

  # We only raise if the strings have same values & prefixes, even if they would eventually evaluate to the same value
  {"one", "two", u"one"}  # FN (ok)
  {"one", "two", r"one"}  # FN (ok)
  {"on" "e", "two", "o" "ne"} # Noncompliant
  {"on" r"e", "two", "on" r"e"} # Noncompliant
  {"on" r"e", "two", "o" r"ne"} # FN (ok)

  # No issue on f-strings to avoid any risk of FP
  p = 1
  {f"one{p}", "two", f"one{p}"} # FN (ok)
  {f"one{p()}", "two", f"one{p()}"} # FN (ok)

def numbers():
  {1, 2, 1}  # Noncompliant
  {1.0, 2.0, 1.0}  # Noncompliant
  {0o1, 0o2, 0O1}  # Noncompliant
  {0x1, 0x3, 0X1}  # Noncompliant
  {0b1, 0o2, 0B1}  # Noncompliant
  {True, False, True}  # Noncompliant

def mixed_types():
  {1.0, 2.0, 0o1}  # Noncompliant
  {1, 2, 1.0}  # Noncompliant
  # True == 1
  {1, 2, True}  # Noncompliant
  # False == 0
  {0, 2, False}  # Noncompliant
  {"1", 1, "2"} # OK


def unpacking_expressions():
  dict = {1}
  {1, *dict} # FN
  {1, *{1}} # FN


def other_cases():
  {None, None}  # Noncompliant
  {(1, "2"), 2, (1, "2")}  # Noncompliant
  {(1, "2"), 2, ("2", 1)}
  # Raises TypeError
  {{"a": 1}, {"a": 1}} # Noncompliant
  {1}
  {frozenset({1}), frozenset({1})} # FN

def repeated_variables(a1, a2, a3):
    {a1, a2, a1}  # Noncompliant
    {a1.b, a2, a1.b}  # Noncompliant
    {func, a2, func}  # Noncompliant

    [{a, a} for a in range(10)]  # Noncompliant

    class MyClass:
        pass

    {MyClass, a2, MyClass}  # Noncompliant
    {MyClass.__doc__, a2, MyClass.__doc__}  # Noncompliant

    {MyClass(), a2, MyClass()}  # OK
    {a1(), a2, a1()}  # OK
    {func(1, 2, 3), a2, func(1, 2, 3)}  # OK

def tuples():
  {(2, bar()), (2, bar())} # function calls
  { (0o10, 'a'), (8, "a")} # Noncompliant
  { (0o10, 'a'), (8, "a", "b")} # OK
  { (1,), 1 } # OK


def large_dict_literal():
  # Accepted FNs to avoid performance issues
  {
  "a", "a", "a", "a", "a", "a", "a", "a", "a", "a",
  "a", "a", "a", "a", "a", "a", "a", "a", "a", "a",
  "a", "a", "a", "a", "a", "a", "a", "a", "a", "a",
  "a", "a", "a", "a", "a", "a", "a", "a", "a", "a",
  "a", "a", "a", "a", "a", "a", "a", "a", "a", "a",
  "a", "a", "a", "a", "a", "a", "a", "a", "a", "a",
  "a", "a", "a", "a", "a", "a", "a", "a", "a", "a",
  "a", "a", "a", "a", "a", "a", "a", "a", "a", "a",
  "a", "a", "a", "a", "a", "a", "a", "a", "a", "a",
  "a", "a", "a", "a", "a", "a", "a", "a", "a", "a",
  "a", "a", "a", "a", "a", "a", "a", "a", "a", "a",
  "a", "b"
  }
