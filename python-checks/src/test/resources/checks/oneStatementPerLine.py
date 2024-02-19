if foo == 'blah': do_blah_thing() #Noncompliant
do_one(); do_two(); do_three() #Noncompliant{{At most one statement is allowed per line, but 3 statements were found on this line.}}

if foo == 'blah':
    do_blah_thing()
do_one()
do_two()
do_three()

import toto; toto.doSomething() #Noncompliant

if foo:
    name = toto
else: name = not_toto #Noncompliant

try: something() #Noncompliant
except OSError: pass #Noncompliant

class SomeClass(object): pass #Noncompliant

def foo(): something() #Noncompliant

if '/c' not in argv: argv += ['/c']  #Noncompliant
def foo(): pass #Noncompliant

def foo():
    try: from pyPgSQL import PgSQL #Noncompliant
    except ImportError:
        pass

def foo():
    try: import gadfly #Noncompliant
    except: return False #Noncompliant
    return True

def dummy_function_impl(): ... # OK

def not_dummyfunction(): something(); ... # Noncompliant

class DummyClassImpl: ... # OK
