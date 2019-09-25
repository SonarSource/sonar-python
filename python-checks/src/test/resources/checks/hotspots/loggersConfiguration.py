import logging
from logging import Logger, Handler, Filter
from logging.config import fileConfig, dictConfig

logging.basicConfig()  # Noncompliant {{Make sure that this logger's configuration is safe.}}

logging.disable()      # Noncompliant


def update_logging(logger_class):
    foo().setLoggerClass(logger_class)    # OK
    logging.otherFn(logger_class)         # OK
    logging.setLoggerClass(logger_class)  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def set_last_resort(last_resort):
    logging.lastResort = last_resort          # Noncompliant
#   ^^^^^^^^^^^^^^^^^^
    foo().lastResort = last_resort            # OK
    lastResort()  = last_resort               # OK
    logging.other = last_resort               # OK

def set_last_resert_multiple_assignment(last_resort):
    import logging
    logging.lastResort, foo = last_resort, 2  # Noncompliant

class CustomLogger(Logger):    # Noncompliant
#                  ^^^^^^
    pass

class CustomLogger(foo()):    # OK
    pass

class CustomLogger():         # OK
    pass

class CustomLogger(OtherLogger):        # OK
    pass

class CustomLogger(logging.OtherClass): # OK
    pass

class CustomHandler(Handler):  # Noncompliant
    pass


class CustomFilter(Filter):    # Noncompliant
    pass


def update_config(path, config):
    fileConfig(path)    # Noncompliant
    dictConfig(config)  # Noncompliant
