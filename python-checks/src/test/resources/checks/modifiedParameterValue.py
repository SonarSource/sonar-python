import collections

def list_modified(param=list()):  # Noncompliant {{Change this default value to "None" and initialize this parameter inside the function/method.}}
#                 ^^^^^^^^^^^^
    param.append('a')
#   ^^^^^^^^^^^^<  {{The parameter is modified.}}
    return param

def set_modifed(param = set()): # Noncompliant
    param.add(42)

def dict_modified(param={}): # Noncompliant
    param.pop()

def deque_modified(param = collections.deque()): # Noncompliant
    param.popleft()

def user_list_modified(param = collections.UserList()): # Noncompliant
    param.append()

def user_dict_modified(param = collections.UserDict()): # Noncompliant
    param.clear()

def chain_map_modified(param = collections.ChainMap()): # Noncompliant
    param.clear()

def counter_modified(param = collections.Counter()): # Noncompliant
    param.subtract()

def ordered_dict_modified(param = collections.OrderedDict()): # Noncompliant
    param.move_to_end()

def default_dict_modified(param1 = collections.defaultdict(), param2 = collections.defaultdict()): # Noncompliant 2
    param1.__getitem__()
    print(param2[1])


def common_methods(param1=list(), param2=list(), param3=list()): # Noncompliant 3
    param1.__delitem__(1)
    param2.__setitem__(0, 42)
    param3.__iadd__(42)

def del_item(param=list()): # Noncompliant
    del param[1]

def set_item(param=list()): # Noncompliant
    param[1] = 42

def compound(param1=list(), param2=list()): # Noncompliant
    #        ^^^^^^^^^^^^^
    param1[1] += 42
    param2 += param1

def compliant(param1=list(), param2=collections.defaultdict(), param3='str', param4, param5=list(), param6='str'):
    print(param1[1])
    print(param1[param2])
    del param3[1]
    param4.append()
    [] += param5
    param6 = 'otherstr'

class A: pass

def attribute_set(param1=A(), param2=None): # Noncompliant
#                 ^^^^^^^^^^
  param1.attr = 42
  param2.attr = 42

def assignment(param1=list()):
    param1 = [1,2,3]

class B:
  def referenced_outside(self, param1 = list(), param2=list()): # Noncompliant 2
      self.l = param1
      self[l] = param2
  def referenced_outside_secondary(self, param1 = list(), param2=list()): # Noncompliant {{Change this default value to "None" and initialize this parameter inside the function/method.}}
#                                        ^^^^^^^^^^^^^^^
      self.l = param1
#     ^^^^^^^^^^^^^^^< {{The parameter is stored in another object.}}
  def referenced_outside_2(self, param = list()): # OK, not on self
      global g_var
      g_var = param
  def maybe_reference_outside(other, param=list()):
      other.l = param
  def not_reference_outside(self, param = list()):
      l = param


def wrapper():
    def nested_fn(param=list()): # OK
        param.append(42)

def using_cache(cache=list(), memo=set()):
    cache[42] = 0
    memo.add(42)
