# x = 1

# Noncompliant@-2

# Noncompliant@+2 {{Remove this commented out code.}} 

# x += 1

# ^[sc=1;ec=8]@-1

x = 1
#Noncompliant@+2

#x = 3

self.setContent('''
"A module string"

from foo import bar

def testFunction():
    "A doc string"
    a = 1 + 2
    return a
''')

something = '''
from foo import bar

def testFunction():
    "A doc string"
    a = 1 + 2
    return a
'''
# Noncompliant@+2

# if x != 2:
#     print('Hello!')
# else:
#     x = 0

# this is not a code

#

# comment/message

# comment-message

## comment

############

# fd is already in place

# IReactorSSL (sometimes, not implemented)

# override in subclasses

# just in case

# call and check

# new in 8.0

# PyPAM is missing

# "subCommands is a list of 4-tuples of (command name, command

# Kind is "user" or "group"

# failure or None

#defined in run() and _run()

####################
# CORE             #
####################

# a code library for the dynamic creation of images

# if x == 4:
print("x is 4")

# 'string'
'''
this is a
multiline comment
'''
# # Check that the parsed result does a round trip to the same format

# IBuildRequestStatus
# ISlaveStatus

# Noncompliant@+2

#     return a

# Noncompliant@+1
u'''
return a
'''

# Noncompliant@+1 
'''
return a
'''
#^[sc=1;ec=3;el=+2]@-2

x = u'''
return a
'''

'''

'''

'''
# comment
'''

# TODO: something

# Noncompliant@+1
a = 1  # a = 1


# Ignore line matching the exception regex

# fmt: off
# pylint: disable=line-too-long

# pyformat: enable
# pyformat: disable

# Only a full match is excluded
# Noncompliant@+2

# abcfmt: off
# abcpylint: disable=line-too-long


# Databricks notebooks
# COMMAND ----------

# MAGIC %md 
# MAGIC ## Alter tables

# COMMAND ----------
