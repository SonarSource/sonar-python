class Foo:
    @ignore_me
    def compliant_1(x):
        pass

    @ignore_me_as_well
    def compliant_2(x):
        pass

    @do_not_ignore_me
    def non_compliant_1(x): # FN "do_not_ignore_me" contains a ignored decorator "ignore_me"
        pass

    @abstractmethod
    def non_compliant_2(x): # Noncompliant
        pass
