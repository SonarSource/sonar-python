try:
    foo()
except SystemExit:  # Noncompliant {{Reraise this exception to stop the application as the user expects}}
      #^^^^^^^^^^
    pass

try:
    foo()
except:  # Noncompliant {{Specify an exception class to catch or reraise the exception}}
    pass

try:
    foo()
except BaseException: # Noncompliant {{Catch a more specific exception or reraise the exception}}
      #^^^^^^^^^^^^^
    pass

try:
    foo()
except ValueError:
    pass
except SystemExit: # Noncompliant {{Reraise this exception to stop the application as the user expects}}
      #^^^^^^^^^^
    pass

try:
    foo()
except SystemExit:  # Noncompliant
    raise ValueError()

try:
    foo()
except SystemExit as ex: # Noncompliant
      #^^^^^^^^^^
    try:
        bar()
    except SystemExit as ex2:
        raise ex2

try:
    foo()
except SystemExit as ex:
    try:
        bar()
    except SystemExit:
        raise # FN

try:
    foo()
except: # Noncompliant
    # This should not be compliant as SystemExit was not handled and the except clause below is unreachable
    pass
except SystemExit:
    raise

try:
    foo()
except (KeyboardInterrupt, SystemExit) as e:
    raise e
except:
    pass

try:
    foo()
except (KeyboardInterrupt, SystemExit) as e:
    raise e
except BaseException:
    pass

try:
    foo()
except (KeyboardInterrupt, SystemExit): # Noncompliant
                          #^^^^^^^^^^
    pass

# using the new python 3.14 syntax
try:
    foo()
except KeyboardInterrupt, SystemExit: # Noncompliant
                         #^^^^^^^^^^
    pass
except ValueError, TypeError: 
    pass


try:
    foo()
except (KeyboardInterrupt, SystemExit) as e:
    raise e

try:
    foo()
except (ValueError, KeyboardInterrupt):
    pass
except: # Noncompliant {{Specify an exception class to catch or reraise the exception}}
    pass

try:
    foo()
except (ValueError, KeyboardInterrupt) as e:
    raise e
except: # Noncompliant {{Specify an exception class to catch or reraise the exception}}
    pass

try:
    foo()
except (ValueError, KeyboardInterrupt) as e:
    raise e

try:
    foo()
except SystemExit as e:
    raise e
except KeyboardInterrupt:
    raise

try:
    foo()
except BaseException as e:
    raise e

try:
    foo()
except BaseException:
    raise
except:
    raise

try:
    foo()
except:
    raise

try:
    foo()
except ThereIsNoExceptionWithThisName:
    raise
except AndThereIsNoExceptionLikeThisEither as e:
    raise e
except ValueError:
    raise ThisIsTheThirdNonExistingException()
except NameError:
    raise x
except AnotherErrorKind and ThisErrorKind:
    raise

# Do not raise an issue in the case somebody calls sys.exit
import sys

try:
    foo()
except:
    sys.exit(1)

# If somebody calls sys.exc_info, just assume they know what they are doing
try:
    foo()
except:
    info = sys.exc_info()

try:
    foo()
except:
    raise SystemExit()

try:
    foo()
except (KeyboardInterrupt, AnotherException):
    sys.exit(1)

