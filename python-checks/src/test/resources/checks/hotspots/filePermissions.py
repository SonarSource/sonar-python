import os, stat
unknown.call()

# umask

## Octal
os.umask(0o777)  # Compliant
os.umask(0o007)  # Compliant

os.umask(*other)  # Compliant

os.umask(0o775)  # Noncompliant {{Make sure this permission is safe.}}
#        ^^^^^
os.umask(0o000)  # Noncompliant
my_val = 0o775
os.umask(my_val) # Noncompliant

multiple_assigned = 0o776
multiple_assigned = 0o775
os.umask(multiple_assigned) # FN

## Decimal
os.umask(511)  # Compliant
os.umask(7)    # Compliant
os.umask(0, other)    # Compliant

os.umask(509)  # Noncompliant
os.umask(0)    # Noncompliant

# chmod functions
os.chmod("/tmp/fs", 0o770)  # Compliant
os.chmod("/tmp/fs", 0770)  # Compliant (Python 2 syntax)
os.chmod(mode=0o777)  # Compliant
os.chmod(mode=unknown.something, path="/tmp/fs")  # Compliant
os.chmod("/tmp/fs", "something")  # Compliant
os.chmod("/tmp/fs", 0o777)  # Noncompliant
os.chmod(mode=0o777, path="/tmp/fs")  # Noncompliant

os.lchmod("/tmp/fs", 0o770)  # Compliant
os.lchmod("/tmp/fs", 0o777)  # Noncompliant

os.fchmod("/tmp/fs", 0o770)  # Compliant
os.fchmod("/tmp/fs", 0o777)  # Noncompliant

## Compliant values for the "mode“ parameter

### Octal
os.chmod("/tmp/fs", 0o0770)  # Compliant -rwxrwx---
os.chmod("/tmp/fs", 0o770)   # Compliant -rwxrwx---
os.chmod("/tmp/fs", 0o70)    # Compliant ----rwx---
os.chmod("/tmp/fs", 0o0)     # Compliant ----------

os.chmod("/tmp/fs", 0o0777)  # Noncompliant
os.chmod("/tmp/fs", 0o0551)  # Noncompliant
os.chmod("/tmp/fs", 0o0007)  # Noncompliant
os.chmod("/tmp/fs", 0o007)   # Noncompliant
os.chmod("/tmp/fs", 0o07)    # Noncompliant
os.chmod("/tmp/fs", 0o7)     # Noncompliant

### Decimal
os.chmod("/tmp/fs", 32)   # Compliant ----r-----
os.chmod("/tmp/fs", 256)  # Compliant -r--------
os.chmod("/tmp/fs", 0)    # Compliant ----------

os.chmod("/tmp/fs", 4)     # Noncompliant
os.chmod("/tmp/fs", 260)   # Noncompliant

### Constants
os.chmod("/tmp/fs", stat.S_IRUSR) # Compliant -r--------
os.chmod("/tmp/fs", stat.S_IWUSR) # Compliant --w-------
os.chmod("/tmp/fs", stat.S_IXUSR) # Compliant ---x------
os.chmod("/tmp/fs", stat.S_IRWXU) # Compliant -rwx------
os.chmod("/tmp/fs", stat.S_IRGRP) # Compliant ----r-----
os.chmod("/tmp/fs", stat.S_IWGRP) # Compliant -----w----
os.chmod("/tmp/fs", stat.S_IXGRP) # Compliant ------x---
os.chmod("/tmp/fs", stat.S_IRWXG) # Compliant ----rwx---
os.chmod("/tmp/fs", stat.S_IRWXU | stat.S_IRWXG)  # Compliant -rwxrwx---
os.chmod("/tmp/fs", stat.S_IRGRP | stat.S_IRUSR)  # Compliant -r--r-----

os.chmod("/tmp/fs", stat.S_IROTH)  # Noncompliant
os.chmod("/tmp/fs", stat.S_IWOTH)  # Noncompliant
os.chmod("/tmp/fs", stat.S_IXOTH)  # Noncompliant
os.chmod("/tmp/fs", stat.S_IRWXO)  # Noncompliant
os.chmod("/tmp/fs", stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # Noncompliant {{Make sure this permission is safe.}}
os.chmod("/tmp/fs", stat.S_IROTH | stat.S_IRGRP | stat.S_IRUSR)  # Noncompliant {{Make sure this permission is safe.}}
x = stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO
os.chmod("/tmp/fs", x) # Noncompliant
y = stat.S_IRWXO
os.chmod("/tmp/fs", y) # Noncompliant

def no_soe():
  some_val = other_val
  other_val = some_val
  os.chmod("/tmp/fs", other_val) # OK
