def basicPositives(param):
    param is {1: 2}  # Noncompliant {{Replace this "is" operator with "==".}}
    #     ^^ 
    param is dict(a=1)  # Noncompliant {{Replace this "is" operator with "==".}}
    #     ^^
    v = []
    param is v  # Noncompliant {{Replace this "is" operator with "==".}}
    #     ^^

def basicNegatives(param):
    param == {1: 2}
    param != {1, 2, 3}
    param == [1, 2, 3]
    param == dict(a=1)
    mylist = []
    param == mylist

def comprehensions(p):
  p is { x: x for x in range(10) } # Noncompliant {{Replace this "is" operator with "==".}}
  # ^^
  p is not [ x for x in range(10) ] # Noncompliant {{Replace this "is not" operator with "!=".}}
  # ^^^^^^
  p is { x for x in range(10) } # Noncompliant {{Replace this "is" operator with "==".}}
  # ^^

def escape_00(p):
  d = dict(a=100)
  d is p # Noncompliant {{Replace this "is" operator with "==".}}
  # ^^

def escape_01(p):
  d = dict(a=100)
  save("something", d)
  e = load("something")
  e is d # ok, `d` could have escaped, and actually ended up in `e`

def escape_02(p):
  d = list(1, 2, 3)
  e = d
  e is not d # ok, `d` really did end up in `e`.

def escape_03(p):
  x = dict(a = 1)
  #   ^^^^^^^^^^^> 1 {{This expression creates a new object every time.}}
  x is p # Noncompliant {{Replace this "is" operator with "==".}}
  # ^^
  p is x # Noncompliant {{Replace this "is" operator with "==".}}
  # ^^

def coverage():
  a[b] is b[a]

x is y # repeatedly checked: required for coverage of an `if (symb != null) {`


# Making sure there are no false positives caused by unpacking assignments
def foo(param):
    a, b = list(param)
    if a is None:  # There is no simple way to know what value is assigned to "a" after unpacking assignment.
        pass

    x, y = list(p)
    if x is p: # that shouldn't report anything
      pass


# Clashes with S5727 and S3403
# https://jira.sonarsource.com/browse/SONARPY-507
def clashS5727Test(p):
    a = list()
    if a is None:  # Only S5727 should raise
        pass

    b = 42
    if None is not b: # for coverage
      pass

    None is p
    p is None

def clashS3403Test():
    a = list()
    if a is "":  # Only S3403 should raise an issue
        pass


# rest is mutably borrowed from `expected-issues/python/src/RSPEC_5796`.
def literal_comparison(param):
    # dict
    param is {1: 2, 3: 4}  # Noncompliant {{Replace this "is" operator with "==".}}
    #     ^^
    # list
    param is [1, 2, 3]  # Noncompliant {{Replace this "is" operator with "==".}}
    #     ^^
    # set
    param is {1, 2, 3}  # Noncompliant {{Replace this "is" operator with "==".}}
    #     ^^

    # issues are also raised for "is not"
    param is not {1: 2, 3: 4}  # Noncompliant {{Replace this "is not" operator with "!=".}}
    #     ^^^^^^
    param is not [1, 2, 3]  # Noncompliant {{Replace this "is not" operator with "!=".}}
    #     ^^^^^^
    param is not {1, 2, 3}  # Noncompliant {{Replace this "is not" operator with "!=".}}
    #     ^^^^^^

    # issues are raised when literals are on the right or on the left of the operator
    {1} is param  # Noncompliant {{Replace this "is" operator with "==".}}
    #   ^^


def builtin_constructor(param):
    """We can't raise on every constructor because some might always return the same object (singletons).
    However we can raise on builtin functions because we know that they won't return the same value.
    """
    # dict
    param is dict(a=2, b=3)  # Noncompliant {{Replace this "is" operator with "==".}}
    #     ^^
    # list
    param is list({4, 5, 6})  # Noncompliant {{Replace this "is" operator with "==".}}
    #     ^^
    # set
    param is set([1, 2, 3])  # Noncompliant {{Replace this "is" operator with "==".}}
    #     ^^
    # complex
    param is complex(1, 2)  # Noncompliant {{Replace this "is" operator with "==".}}
    #     ^^

glob = 5
def variable(param):
    mylist = []
    param is mylist  # Noncompliant {{Replace this "is" operator with "==".}}
    #     ^^

    referenced = []
    def referencing():
        nonlocal referenced
        referenced = param
        param is referenced  # No issue

    referencing()
    param is referenced  # No issue

    reassigned = []
    reassigned = param
    param is reassigned  # No issue

    global glob
    param is glob  # No issue (this might/should break once symbols are resolved outside of function definitions)


def locations_and_messages(param):
    param is {1: 2, 3: 4}  # Noncompliant {{Replace this "is" operator with "==".}}
#         ^^

    # {1: 2, 3: 4} is not complex(1, 2)  # Originally marked as "noncompliant", but clashes with RSPEC-3403

    mylist = [] # secondary should be here
    param is mylist  # Noncompliant {{Replace this "is" operator with "==".}}
#         ^^

