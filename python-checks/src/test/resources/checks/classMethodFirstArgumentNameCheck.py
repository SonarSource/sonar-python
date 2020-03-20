class A(object):
  @classmethod
  def first_param_wrong_name(bob, a, b): ... # Noncompliant

  @classmethod
  def cls_ok(cls, height, width): ...  # OK

  @classmethod
  def mcs_ok(mcs, height, width): ...  # OK

  @classmethod
  def self_first_param(self, a, b): ... # Noncompliant

  def __init_subclass__(bob, height, width): ...  # Noncompliant

  def __class_getitem__(bob, a, b): ... # Noncompliant

  def __new__(bob, a, b): ... # Noncompliant

  @random.decorator
  def some_method(a, b): ... # OK

  @classmethod.unrelated
  def unpacking(a, b): ... # OK

  @unrelated_classmethod
  def unpacking(a, b): ... # OK

  @classmethod
  def unpacking(*args): ... # OK

  @classmethod
  def cls_keyword_only(*, cls, x): ... # handled by S5719

  @classmethod
  def no_cls_keyword_only(*, a, b): ... # handled by S5719

  @classmethod
  def no_parameters(): ... # handled by S5719

  @classmethod
  def tuple_first_param((a, b), c): ... # This would be a bug to be handled by S5719


def some_function(): ... # OK
