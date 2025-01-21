from modB import B

class A(B):
  @classmethod
  def self_first_param(self, a, b): ... # Noncompliant
