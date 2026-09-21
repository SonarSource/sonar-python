# A module-level wrapper that aliases and shadows the builtin "reversed".
# "_reversed" captures the builtin at runtime before the shadowing "def reversed" rebinds the name.
_reversed = reversed


def reversed(*args, **kwargs):
    # The wrapper forwards its arguments via unpacking (*args, **kwargs), so there is no
    # positional argument to inspect and no false positive is raised here.
    print(args, kwargs)
    return _reversed(*args, **kwargs)


# FN: because "reversed" is reassigned at module scope, "_reversed" is inferred as the wrapper
# function rather than the builtin, so this genuine "reversed(set)" bug is not flagged.
_reversed({1, 2, 3})
_reversed([1, 2, 3])

# "reversed" now refers to the user-defined wrapper, not the builtin -> correctly not flagged.
reversed({1, 2, 3})
