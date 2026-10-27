def public(cls):
    return cls


def doctest_depends_on(**dependencies):
    def decorate(cls):
        cls.dependencies = dependencies
        return cls
    return decorate
