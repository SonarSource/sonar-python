correct_lamda = lambda p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13: p1

incorrect_lamda = lambda p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14=42: p1 # Noncompliant {{Lambda has 14 parameters, which is greater than the 13 authorized.}}
#                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

empty_params_lambda = lambda: True

def correct_function(
    p1, p2, p3, p4, p5, p6, p7=None,
    p8=None, p9=None, p10=None,
    p11=None, p12=None, p13=None):
	...

def incorrect_function(
    p1, p2, p3, p4, p5, p6,  # Noncompliant {{Function "incorrect_function" has 14 parameters, which is greater than the 13 authorized.}}
#   ^[el=+3;ec=17]
    p7, p8, p9, p10, p11,
    p12, p13, p14):
	...

def empty_params_function(): ...

class MyClass:
	def correct_method(self, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13):
		...

	@classmethod
	def correct_class_method(cls, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13):
	  ...

	@staticmethod
	def incorrect_static_method(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14):  # Noncompliant {{Method "incorrect_static_method" has 14 parameters, which is greater than the 13 authorized.}}
#	                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	  ...

	def incorrect_method(self, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14): # Noncompliant {{Method "incorrect_method" has 14 parameters, which is greater than the 13 authorized.}}
		...

def star_parameter(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, *p14): # Noncompliant
	...

def method_with_keyword_only_params(p1, p2, p3, *, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13):
  ...

def method_with_positional_only_params(p1, p2, p3, /, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13):
  ...


class SomeBase():
  def incorrect_method(
      self, p1, p2, p3, p4, p5, p6,  # Noncompliant {{Method "incorrect_method" has 14 parameters, which is greater than the 13 authorized.}}
      p7, p8, p9, p10, p11,
      p12, p13, p14):
  	...

  def initially_correct_method(
      self, p1, p2, p3):
    ...

class OtherBase():
  def initially_correct_method(
      self, p1, p2, p3, p4, p5, p6,  # Noncompliant {{Method "initially_correct_method" has 14 parameters, which is greater than the 13 authorized.}}
      p7, p8, p9, p10, p11,
      p12, p13, p14):
    ...

  initially_not_a_method = 42

class SomeChild(SomeBase):
  def incorrect_method(
      self, p1, p2, p3, p4, p5, p6,  # OK (inherited)
      p7, p8, p9, p10, p11,
      p12, p13, p14):
  	...

  def initially_correct_method(
      self, p1, p2, p3, p4, p5, p6,  # Noncompliant {{Method "initially_correct_method" has 14 parameters, which is greater than the 13 authorized.}}
      p7, p8, p9, p10, p11,
      p12, p13, p14):
    ...

class OtherChild(OtherBase, SomeBase):
  def initially_correct_method(
      self, p1, p2, p3, p4, p5, p6,  # OK (inherited from OtherBase)
      p7, p8, p9, p10, p11,
      p12, p13, p14, p15):
    ...
  def initially_not_a_method(
      self, p1, p2, p3, p4, p5, p6,  # Noncompliant {{Method "initially_not_a_method" has 15 parameters, which is greater than the 13 authorized.}}
      p7, p8, p9, p10, p11,
      p12, p13, p14, p15):
    ...

if cond:
  def ambiguous_symbol(): ...
else:
  class ambiguous_symbol():
    def incorrect_method(
        self, p1, p2, p3, p4, p5, p6,  # Noncompliant {{Method "incorrect_method" has 14 parameters, which is greater than the 13 authorized.}}
        p7, p8, p9, p10, p11,
        p12, p13, p14):
      ...

class ChildOfAmbiguous(ambiguous_symbol):
  # FP (SONARPY-656)
  def incorrect_method(
      self, p1, p2, p3, p4, p5, p6,  # Noncompliant {{Method "incorrect_method" has 14 parameters, which is greater than the 13 authorized.}}
      p7, p8, p9, p10, p11,
      p12, p13, p14):
    ...
