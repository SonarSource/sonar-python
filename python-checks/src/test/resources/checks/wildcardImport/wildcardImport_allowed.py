try:
    from foo.bar import *  # OK - we are handling possibly missing third-party libraries
except Exception:
    from baz import *  # OK

try:
    from pyrepl.readline import *  # OK
except ImportError:
    import sys
    if sys.platform == 'win32':
        raise ImportError
    raise

import warnings

warnings.warn(("warning"), category=DeprecationWarning, stacklevel=2)
from some_module import *  # OK - backward compatibility

__all__ = [ "FOO", "BAR", "BAZ" ] # Assignments to __all__ should count as an allowed statement
