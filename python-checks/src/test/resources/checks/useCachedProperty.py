import functools
from enum import Enum
from functools import cache, cached_property, lru_cache


class WithLruCache:
    @lru_cache  # Noncompliant {{Use "@cached_property" instead of "@lru_cache" or "@cache" on this method.}}
   #^^^^^^^^^^
    def fib(self) -> int:
        # long computation
        return self.scale

    @lru_cache()  # Noncompliant
   #^^^^^^^^^^^^
    def once(self):
        return 1

    @lru_cache(maxsize=128)  # Noncompliant
   #^^^^^^^^^^^^^^^^^^^^^^^
    def with_maxsize(self):
        return 1

    @lru_cache(maxsize=None)  # Noncompliant
   #^^^^^^^^^^^^^^^^^^^^^^^^
    def unbounded(self):
        return 1

    @lru_cache  # Noncompliant
   #^^^^^^^^^^
    def positional_only(self, /):
        return 1


class WithCache:
    @cache  # Noncompliant {{Use "@cached_property" instead of "@lru_cache" or "@cache" on this method.}}
   #^^^^^^
    def total(self) -> int:
        return 1


class WithQualifiedName:
    @functools.lru_cache  # Noncompliant
   #^^^^^^^^^^^^^^^^^^^^
    def fib(self):
        return 1

    @functools.lru_cache(maxsize=32)  # Noncompliant
   #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    def bounded(self):
        return 1

    @functools.cache  # Noncompliant
   #^^^^^^^^^^^^^^^^
    def total(self):
        return 1


class WithOtherDecorators:
    @property
    @lru_cache  # Noncompliant
   #^^^^^^^^^^
    def value(self):
        return 1


class StillInstanceMethod:
    @lru_cache  # Noncompliant
   #^^^^^^^^^^
    def renamed(this):
        return 1


class Compliant:
    @lru_cache
    def fib(self, n):
        return n

    @cache
    def with_default(self, n=1):
        return n

    @lru_cache
    def keyword_only(self, *, n):
        return n

    @cache
    def varargs(self, *args):
        return args

    @lru_cache
    def kwargs(self, **kwargs):
        return kwargs

    @cached_property
    def already_cached(self):
        return 1

    @functools.cached_property
    def already_cached_qualified(self):
        return 1

    @lru_cache
    async def async_total(self):
        return 1

    def plain(self):
        return 1

    @staticmethod
    @lru_cache
    def static_cached():
        return 1

    @classmethod
    @cache
    def class_cached(cls):
        return 1

    @lru_cache
    def __init_subclass__(cls, **kwargs):
        return None

    @cache
    def __class_getitem__(cls, item):
        return cls

    def nested(self):
        @lru_cache
        def helper(value):
            return value
        return helper(1)


class Color(Enum):
    RED = 1

    @lru_cache
    def label(self):
        return self.name


@lru_cache
def free_function():
    return 1


@cache
def free_function_with_arg(value):
    return value


class EmptyOrVarargsOnly:
    def safe_property(func):
        return func

    @safe_property
    def info():
        return {}

    @lru_cache
    def cached_without_params():
        return 1

    @lru_cache
    def only_varargs(*args):
        return args

    @cache
    def only_kwargs(**kwargs):
        return kwargs
