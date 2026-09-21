def noncompliant_builtins():
    reversed({1, 2, 3, 4})          # Noncompliant {{Change the argument of this "reversed()" call to a reversible object (one implementing "__reversed__", or both "__len__" and "__getitem__").}}
#            ^^^^^^^^^^^^
    reversed(frozenset({1, 2, 3}))  # Noncompliant
    reversed(5)                     # Noncompliant
    reversed(3.14)                  # Noncompliant


def noncompliant_custom_class():
    class NotReversible:
        pass

    reversed(NotReversible())  # Noncompliant

    class OnlyLen:
        def __len__(self):
            return 0

    reversed(OnlyLen())  # Noncompliant

    class OnlyGetItem:
        def __getitem__(self, index):
            return index

    reversed(OnlyGetItem())  # Noncompliant


def noncompliant_iterators():
    reversed(map(str, [1, 2, 3]))   # Noncompliant
    reversed(enumerate([1, 2, 3]))  # Noncompliant
    reversed(x for x in range(3))   # Noncompliant
    reversed((x for x in range(3)))  # Noncompliant
    reversed({x for x in range(3)})  # Noncompliant


def compliant_builtins():
    reversed([1, 2, 3, 4])
    reversed((1, 2, 3, 4))
    reversed("abcd")
    reversed(b"abcd")
    reversed({"a": 1, "b": 2})
    reversed(range(10))
    reversed([x for x in range(3)])  # list comprehension is a list -> reversible
    reversed({k: 0 for k in range(3)})  # dict comprehension is a dict -> reversible


def compliant_custom_class():
    class WithReversed:
        def __reversed__(self):
            return iter([])

    reversed(WithReversed()) # Compliant

    class Sequence:
        def __getitem__(self, index):
            return index

        def __len__(self):
            return 0

    reversed(Sequence())  # Compliant


def compliant_inheritance():
    class MyList(list):
        pass

    reversed(MyList())  # Compliant: inherits list's __reversed__


def unknown_type(param):
    reversed(param)  # Compliant type of param is unknown


def not_the_builtin_reversed():
    def reversed(x):
        return x

    reversed({1, 2, 3})  # Compliant not the builtin reversed


def aliased_reversed():
    _reversed = reversed
    _reversed({1, 2, 3})  # Noncompliant
    _reversed([1, 2, 3])


def genuine_union_is_suppressed(cond):
    # We suppress if any candidate is reversible. For a genuine union that holds at
    # least one reversible member, the check does not flag it. The 'set' path below would fail
    # at runtime, this FN is accepted to remove the far more common flow-insensitive FPs.
    mixed = [1, 2, 3] if cond else {1, 2, 3}
    reversed(mixed)  # FN: 'mixed' is (list | set); the set branch is not reversible


def optional_list_early_return(stack: list | None):
    if stack is None:
        return
    for s in reversed(stack):  # 'stack' is a list here (inferred list | None)
        print(s)


def optional_list_default(frames: list | None = None):
    if frames is None:
        frames = []
    return list(reversed(frames))  # 'frames' is a list here (inferred list | None)


def no_argument():
    reversed()  # Compliant
