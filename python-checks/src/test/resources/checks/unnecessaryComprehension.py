
def case1():
  iterable_single = [1, 2, 3, 2]
  list_comp = [x for x in iterable_single] # Noncompliant
  set_comp = {x for x in iterable_single} # Noncompliant
  list_comp = list(x for x in iterable_single) # Noncompliant
  set_comp = set(x for x in iterable_single) # Noncompliant

  iterable_pairs = [('a', 1), ('b', 2)]
  dict_comp = {k: v for k, v in iterable_pairs} # Noncompliant

def case2():
  iterable_single = [1, 2, 3, 2]
  list_comp = [y for x in iterable_single]
  tuple = tuple(x for x in iterable_single if not something(x))
  list_comp = [x + 1 for x in iterable_single]
  set_comp = {x + 1 for x in iterable_single}
  list_comp = list(x + 1 for x in iterable_single)
  set_comp = set(x + 1 for x in iterable_single)

  iterable_pairs = [('a', 1), ('b', 2)]
  dict_comp = {k: v + 1 for k, v in iterable_pairs}
  dict_comp = {k: v for k, v.x in iterable_pairs}
  dict_comp = {k: v for k.x, v in iterable_pairs}
  dict_comp = {something(k): v for k, v in iterable_pairs}
  dict_comp = {k: v for k, v in iterable_pairs if something(k)}
  dict_comp = {k: v1 for k, v2 in iterable_pairs}
  dict_comp = {k1: v for k2, v in iterable_pairs}
