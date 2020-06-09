def func():
    return "item1" "item2"  # Noncompliant {{Merge these implicitly concatenated strings; or did you forget a comma?}}

def func():
    return "item1", "item2" # OK

def list_literals():
  ['1' '2']  # Noncompliant {{Merge these implicitly concatenated strings; or did you forget a comma?}}
  ['1' '2', '3']  # Noncompliant
  ['1', '2' '3']  # Noncompliant
  ['1', '2' '3', '4']  # Noncompliant
  ['a', 'b'  # Noncompliant {{Add a "+" operator to make the string concatenation explicit; or did you forget a comma?}}
   'c']
  ["1",
   "2"  # Noncompliant
   "3",
   "a very very very"  # Noncompliant
   "very very long string",
   "4"]

def sets():
  {'1' '2'}  # Noncompliant
  {'1' '2', '3'}  # Noncompliant
  {'1', '2' '3'}  # Noncompliant

def formatting():
  ["1",
   "2",
   "3",
   "explicit %s"
   " would fail here" % "string concatenation",
   "explicit {}"
   " would fail here".format("string concatenation"),
   "4"]

def calls():
  print('1', '2' '3')  # Noncompliant
  print('Some long'
  ' message')  # OK
  ('a tuple',
    call('a long'
    'string'))

def tuples():
  # Note that the following example isn't actually a tuple.
  ('1' '2')  # Noncompliant
  ('1' '2', '3')  # Noncompliant
  ('1', '2' '3')  # Noncompliant
  ('a', 'b'  # Noncompliant
   'c')

def prefixes():
  f'1' '2'  # Ok. f-string and string
  F'1' '2'  # Ok. prefixes are case insensitive
  F'1' f'2'  # Noncompliant
  F'1' fr'2'  # Ok
  u'1' '2' # Ok
  (u"1" u"2")  # Noncompliant
  {r'''1''' r'''2'''}  # Noncompliant
  [b'A' b'B']  # Noncompliant

def different_quotes():
  ["""1""" """2"""]  # Noncompliant
  "1" '2' # Ok
  "1" """
  2""" # Ok. Even if strange

def edge_cases():
  my_string = ('a'  # Ok. Even if it is suspicious (this is not a tuple)
      'b')
  ('1 \
   2') # OK
  # Noncompliant@+1
  ('1' '2 \
   3')


def exceptions():
  ["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    "b"]
  ["a\n"
    "b"]
  ["a"
    "\nb"]
  ["a "
    "b"]
  ["a"
    " b"]
  ["a,"
    "b"]
  ["a"
    ",b"]
