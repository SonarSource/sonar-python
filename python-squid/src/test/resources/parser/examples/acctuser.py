#!/usr/bin/python

# Exercise the Accounts class.

# Get the acct module, which contains the Accounts class.
from acct import Accounts

# Print all the existing accounts and values.
def praccts(ao):
    "Print all accounts."
    for n in ao.list():
        print '%-15s %4d' % (n + ':', ao.value(n))

# Create the account set with 4000 in nimrod, then create several other
# accounts starting with 400 taken from nimrod.
act = Accounts('nimrod', 4000)
for n in ['bogus', 'dingle', 'fredburt', 'milhouse']:
    act.newacct(n)
    act.transfer('nimrod', n, 400)
print '**** Initially ****'
praccts(act)
print

# Some random operations.
act.transfer('dingle', 'bogus', 45)
act.transfer('fredburt', 'milhouse', 155)
act.transfer('fredburt', 'dingle', 221)
act.close('fredburt', 'nimrod')

print '**** Finally ****'
praccts(act)
print
