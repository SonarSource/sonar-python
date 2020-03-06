try:
    pass
except SystemExit:  # Noncompliant {{Reraise this exception to stop the application as the user expects}}
      #^^^^^^^^^^
    pass
except KeyboardInterrupt as e:  # Noncompliant {{Reraise this exception to stop the application as the user expects}}
      #^^^^^^^^^^^^^^^^^
    pass

try:
    pass
except:  # Noncompliant {{Specify an exception class to catch or reraise the exception}}
    pass

try:
    pass
except BaseException: # Noncompliant {{Catch a more specific exception or reraise the exception}}
      #^^^^^^^^^^^^^
    pass

try:
    pass
except ValueError:
    pass
except SystemExit: # Noncompliant {{Reraise this exception to stop the application as the user expects}}
      #^^^^^^^^^^
    pass

try:
    pass
except SystemExit:  # Noncompliant
    raise ValueError()

try:
    open("foo.txt", "r")
except SystemExit as ex: # Noncompliant
      #^^^^^^^^^^
    try:
        open("bar.txt", "r")
    except SystemExit as ex2: # Compliant
        raise ex2

try:
    foo()
except KeyboardInterrupt as ex:
    try:
        foo()
    except SystemExit:
        raise # FN

try:
    pass
except KeyboardInterrupt: # Noncompliant
      #^^^^^^^^^^^^^^^^^
    pass
except: # Noncompliant
    # This should not be compliant as SystemExit was not handled and the except clause below is unreachable
    pass
except SystemExit:
    raise

try:
    pass
except (KeyboardInterrupt, SystemExit): # Noncompliant
    pass

try:
    pass
except (ValueError, KeyboardInterrupt):
    pass
except: # Noncompliant
    pass

try:
    pass
except SystemExit as e:
    raise e
except KeyboardInterrupt:
    raise

try:
    pass
except BaseException as e:
    raise e

try:
    pass
except BaseException:
    raise
except:
    raise

try:
    pass
except:
    raise

try:
    pass
except ThereIsNoExceptionWithThisName:
    raise
except AndThereIsNoExceptionLikeThisEither as e:
    raise e
except ValueError:
    raise ThisIsTheThirdNonExistingException()
