import functools
from enum import Enum, IntEnum, StrEnum, IntFlag, Flag
from functools import cache, lru_cache


class WithLruCacheFromImport:
    @lru_cache  # Noncompliant {{Remove "@lru_cache"/"@cache" from this instance method; the cache retains "self" and can leak memory.}}
   #^^^^^^^^^^
    def squared(self, value):
        return value * value

    @lru_cache()  # Noncompliant
   #^^^^^^^^^^^^
    def cubed(self, value):
        return value ** 3

    @lru_cache(maxsize=128)  # Noncompliant
   #^^^^^^^^^^^^^^^^^^^^^^^
    def with_maxsize(self, value):
        return value

    @lru_cache(maxsize=None)  # Noncompliant
   #^^^^^^^^^^^^^^^^^^^^^^^^
    def unbounded(self, value):
        return value


class WithCacheFromImport:
    @cache  # Noncompliant {{Remove "@lru_cache"/"@cache" from this instance method; the cache retains "self" and can leak memory.}}
   #^^^^^^
    def compute(self, n):
        return n


class WithFunctoolsQualified:
    @functools.lru_cache  # Noncompliant
   #^^^^^^^^^^^^^^^^^^^^
    def squared(self, value):
        return value * value

    @functools.lru_cache(maxsize=32)  # Noncompliant
   #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    def with_args(self, value):
        return value

    @functools.cache  # Noncompliant
   #^^^^^^^^^^^^^^^^
    def compute(self, n):
        return n


class WithOtherDecorators:
    @property
    @lru_cache
    def cached_property_like(self):
        return 42

    @property
    @lru_cache  # Noncompliant
   #^^^^^^^^^^
    def cached_with_arg(self, value):
        return value


@lru_cache
def free_function_lru(value):
    return value * value


@cache
def free_function_cache(value):
    return value


@functools.lru_cache(maxsize=None)
def free_function_qualified(value):
    return value


class WithStaticAndClassMethods:
    @staticmethod
    @lru_cache
    def static_cached(value):
        return value

    @classmethod
    @lru_cache
    def class_cached(cls, value):
        return value

    @lru_cache
    @staticmethod
    def static_cache_outer(value):
        return value

    @lru_cache
    @classmethod
    def class_cache_outer(cls, value):
        return value


class Color(Enum):
    RED = 1
    BLUE = 2

    @lru_cache
    def describe(self):
        return self.name


class Priority(IntEnum):
    LOW = 1
    HIGH = 2

    @cache
    def label(self):
        return str(self.value)

class TextFormat(StrEnum):
    PLAIN = "plain"
    HTML = "html"

    @cache
    def is_plain(self):
        return self is TextFormat.PLAIN


class Permission(Flag):
    READ = 1
    WRITE = 2
    EXECUTE = 4

    @lru_cache
    def can_modify(self) -> bool:
        return bool(self & Permission.WRITE)


class Status(IntFlag):
    INACTIVE = 0
    ACTIVE = 1
    PENDING = 2

    @lru_cache
    def is_active(self) -> bool:
        return bool(self & Status.ACTIVE)

class WithImplicitClassMethods:
    @lru_cache
    def __init_subclass__(cls, **kwargs):
        return None

    @cache
    def __class_getitem__(cls, item):
        return cls


class CompliantInstanceMethods:
    def normal(self, value):
        return value

    def nested_ok(self):
        @lru_cache
        def helper(value):
            return value
        return helper(1)


class OnlySelfIsCompliant:
    @lru_cache
    def once(self):
        return 1

    @cache
    def once_cache(self) -> int:
        return 1

    @functools.lru_cache(maxsize=None)
    def qualified(self):
        return self

    @lru_cache()
    def positional_only(self, /):
        return 1

    @lru_cache  # Noncompliant
   #^^^^^^^^^^
    def still_cached(self, n):
        return n

    @lru_cache  # Noncompliant
   #^^^^^^^^^^
    def varargs(self, *args):
        return args

    @cache  # Noncompliant
   #^^^^^^
    def kwargs(self, **kwargs):
        return kwargs

    @lru_cache  # Noncompliant
   #^^^^^^^^^^
    def keyword_only(self, *, n):
        return n

    @lru_cache  # Noncompliant
   #^^^^^^^^^^
    def with_default(self, n=1):
        return n


class EmptyParameterList:
    def safe_property(func):
        return func

    @safe_property
    def info():
        return {}

    @lru_cache
    def cached_without_params():
        return 1
