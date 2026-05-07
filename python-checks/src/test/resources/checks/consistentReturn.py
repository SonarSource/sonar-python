# simple cases

def no_return_at_all(x):
  print(x)

def return_value_on_all_branches(x):
  if x > 0:
    return x
  else:
    return None

def return_without_value(x): # Noncompliant {{Refactor this function to use "return" consistently.}}
#   ^^^^^^^^^^^^^^^^^^^^
  if x < 0:
    return
#   ^^^^^^ < {{Return without value}}
  return x
# ^^^^^^^^ < {{Return with value}}

def all_returns_have_no_value(x):
  if x > 0:
    print(x)
    return
  print(0)
  return


# implicit return

def return_without_value_and_implicit_return(x):
  if x > 0:
    print(x)
    return
  print(0)

def implicit_return_on_expression_statement(x): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  if x > 0:
    return x
#   ^^^^^^^^ < {{Return with value}}
  print(x)
# ^^^^^^^^ < {{Implicit return without value}}


# implicit return and compound statements

def implicit_return_on_if_statement(x): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  if x > 0:
# ^^ < {{Implicit return without value if the condition is false}}
    return x
#   ^^^^^^^^ < {{Return with value}}

def implicit_return_on_for_statement(x): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  for e in x:
# ^^^ <
    return x
#   ^^^^^^^^ <

def implicit_return_on_while_statement(x): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  while condition(x):
# ^^^^^ <
    x = foo()
    if x > 0:
      return x
#     ^^^^^^^^ <

def while_true(x):
  while True:
    x = get_random_string(32)
    if valid(x):
      return x

def while_with_unknown_condition(x): # Noncompliant
  while condition(x):
    x = get_random_string(32)
    if valid(x):
      return x

def implicit_return_on_def_statement(x): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  if x > 0:
    return x
#   ^^^^^^^^ < {{Return with value}}
  def foo():
# ^^^ < {{Implicit return without value}}
    pass

def implicit_return_on_class_statement(x): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  if x > 0:
    return x
#   ^^^^^^^^ < {{Return with value}}
  class Foo:
    def meth1():
#   ^^^ < {{Implicit return without value}}
      print("meth1")

def implicit_return_on_match_statement(x): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  match(x):
# ^^^^^ < {{Implicit return without value when no case matches}}
    case 0:
      return x
#     ^^^^^^^^ < {{Return with value}}

def match_statement_with_default(x):
  match(x):
    case 0:
      return x
    case _:
      return 42

def match_wildcard_return_or_raise(x):
  match x:
    case 1:
      return 2
    case _:
      raise ValueError("error")

def match_capture_pattern_all_return(x):
  match x:
    case 1:
      return 2
    case 2:
      return 3
    case other:
      return other + 10

def match_wildcard_with_guard(x): # Noncompliant
  match x:
    case 1:
      return 2
    case _ if x > 0:
      return 3

def match_irrefutable_not_last(x):
  match x:
    case _:
      return 1
    case 1:
      return 2

def match_irrefutable_in_middle(x):
  match x:
    case 0:
      return 1
    case _:
      return 2
    case 1:
      pass

def match_group_capture_is_irrefutable(x):
  match x:
    case 0:
      return x
    case (y):
      return y


def match_group_wildcard_is_irrefutable(x):
  match x:
    case 0:
      return x
    case (_):
      return 42


def match_nested_group_is_irrefutable(x):
  match x:
    case 0:
      return x
    case ((y)):
      return y


def match_guarded_group_is_refutable(x): # Noncompliant
  match x:
    case 0:
      return x
    case (y) if x > 0:
      return y


def match_group_literal_is_refutable(x): # Noncompliant
  match x:
    case 0:
      return x
    case (42):
      return 42

# raise / assert

def return_value_or_raise(x):
  if x > 0:
    return x
  raise "error"

def return_value_or_assert(x):
  if x > 0:
    return x
  assert False, "x should be > 0"

def no_secondary_location_on_raise_or_assert(x): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  if x == 0:
    raise "error"
  if x > 99:
    assert False, "unexpected"
  if x < 0:
    return
#   ^^^^^^ <
  return x
# ^^^^^^^^ <


# Limitations (CFG and others)

def bad_cfg(x):
  break

def try_except(x):
  try:
    return foo()
  except Exception:
    return 0

def try_finally(x):
  if x > 0:
    return 0
  try:
    return x
  finally:
    print(x)

def noncompliant_function_containing_try_in_nested_function(p): # Noncompliant
  def nested_function_with_try(x):
    try:
      return x
    finally:
      print(x)
  if p:
    return nested_function_with_try

def with_statement(x):
  with something(x):
    return foo(x)

def with_statement_inside_if(x): # Noncompliant
  if x > 0:
    with something(x):
      return foo(x)

def fp_on_undetected_raise(x): # Noncompliant
  if x > 0:
    return x
  raise_error()

def raise_error():
  raise "error"


# Patterns from ruling analysis

def find_match_in_loop(texts, frag):  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^
  for t in texts:
# ^^^ < {{Implicit return without value if the condition is false}}
    if t is not frag and t.left > frag.right:
      return t
#     ^^^^^^^^ < {{Return with value}}

def multiple_value_returns_with_bare_return(aug_tok1, params):  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  if not aug_tok1:
    return
#   ^^^^^^ < {{Return without value}}
  if aug_tok1.period_final:
    return "REASON_KNOWN_COLLOCATION"
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ < {{Return with value}}
  if aug_tok1.abbr:
    return "REASON_ABBR"
#   ^^^^^^^^^^^^^^^^^^^^ < {{Return with value}}
  return
# ^^^^^^ < {{Return without value}}

def return_only_in_else_branch(run_mode):  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^
  if run_mode == 'v1':
    do_v1_test()
#   ^^^^^^^^^^^^ < {{Implicit return without value}}
  elif run_mode == 'v2':
    do_v2_test()
#   ^^^^^^^^^^^^ < {{Implicit return without value}}
  else:
    return ValueError('Unknown run mode %s' % run_mode)
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ < {{Return with value}}


class MyClass:
  def value_or_none(self): # Noncompliant
#     ^^^^^^^^^^^^^
    if self.condition:
#   ^^ < {{Implicit return without value if the condition is false}}
      return self._value
#     ^^^^^^^^^^^^^^^^^^ < {{Return with value}}

  def always_returns_none_or_value(self):
    if self.condition:
      return self._value
    return None

  @property
  def linked_object(self): # Noncompliant
#     ^^^^^^^^^^^^^
    if self.link_type == "a":
      return self.a
#     ^^^^^^^^^^^^^ < {{Return with value}}
    elif self.link_type == "b":
      return self.b
#     ^^^^^^^^^^^^^ < {{Return with value}}
    elif self.link_type == "c":
      return self.c
#     ^^^^^^^^^^^^^ < {{Return with value}}
    elif self.link_type == "d":
#   ^^^^ < {{Implicit return without value if the condition is false}}
      return self.d
#     ^^^^^^^^^^^^^ < {{Return with value}}


def outer_with_nested(params):
  def handler(cursor): # Noncompliant
#     ^^^^^^^
    if cursor.bindvars is None:
      return
#     ^^^^^^ < {{Return without value}}
    if isinstance(cursor.bindvars, list):
      return [v.getvalue() for v in cursor.bindvars]
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ < {{Return with value}}
    if isinstance(cursor.bindvars, dict):
      return {n: v.getvalue() for n, v in cursor.bindvars.items()}
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ < {{Return with value}}
    raise TypeError("Unexpected bindvars")

  result = run(handler)
  return result


def boundaries(start, end): # Noncompliant
#   ^^^^^^^^^^
  if not isinstance(start, int):
    raise TypeError("expected int")
  if not isinstance(end, int):
    raise TypeError("expected int")
  if start > end:
    start, end = end, start
  if start < end:
# ^^ < {{Implicit return without value if the condition is false}}
    return start, end
#   ^^^^^^^^^^^^^^^^^ < {{Return with value}}


class MiddlewareMixin:
  @staticmethod
  def process_request(request): # Noncompliant
#     ^^^^^^^^^^^^^^^
    if not request.user.is_authenticated:
#   ^^ < {{Implicit return without value if the condition is false}}
      if request.path not in IGNORE_URL:
#     ^^ < {{Implicit return without value if the condition is false}}
        return redirect('/login/')
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^ < {{Return with value}}


class WebhookPlugin:
  def order_fully_paid(self, order, previous_value): # Noncompliant
#     ^^^^^^^^^^^^^^^^
    if not self.active:
      return previous_value
#     ^^^^^^^^^^^^^^^^^^^^^ < {{Return with value}}
    event_type = "ORDER_FULLY_PAID"
    if webhooks := get_webhooks_for_event(event_type):
#   ^^ < {{Implicit return without value if the condition is false}}
      trigger_webhooks_async(webhooks)
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ < {{Implicit return without value}}


class CacheFactory:
  @classmethod
  def factory(cls, backend, ttl): # Noncompliant
#     ^^^^^^^
    if backend == "memory":
      return CacheDict(ttl)
#     ^^^^^^^^^^^^^^^^^^^^^ < {{Return with value}}
    elif backend == "disk":
      return CacheDisk(ttl)
#     ^^^^^^^^^^^^^^^^^^^^^ < {{Return with value}}
    else:
      log.error("unrecognized cache type")
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ < {{Implicit return without value}}


@lru_cache()
def cached_lookup(locale): # Noncompliant
#   ^^^^^^^^^^^^^
  name = get_name(locale)
  if name is not None:
# ^^ < {{Implicit return without value if the condition is false}}
    return load_resource(name)
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^ < {{Return with value}}


@some_hook(flag=True)
def announce_header(**kwargs): # Noncompliant
#   ^^^^^^^^^^^^^^^
  if not session.user or config.DISABLE_CHECK:
    return
#   ^^^^^^ < {{Return without value}}
  if down:
    text = "scheduler not running"
  elif mismatch:
    text = "version mismatch"
  else:
    return
#   ^^^^^^ < {{Return without value}}
  return 'error', text, True
# ^^^^^^^^^^^^^^^^^^^^^^^^^^ < {{Return with value}}


def early_sentinel_value_return(path, conn=None): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  if conn is None:
    conn = get_conn()
  if conn is False:
    return False
#   ^^^^^^^^^^^^ < {{Return with value}}
  for comp in path.split("/"):
# ^^^ < {{Implicit return without value if the condition is false}}
    process(comp)


class SearchHelper:
  def dup_check(self, sequence): # Noncompliant
#     ^^^^^^^^^
    for sc in self.all_shortcuts:
#   ^^^ < {{Implicit return without value if the condition is false}}
      if sc is self.shortcut:
        continue
      for k in sc['keys']:
        if k == sequence:
          return sc['name']
#         ^^^^^^^^^^^^^^^^^ < {{Return with value}}
