# Python classes.

#
# A class representing a set of accounts with a fixed total.
class Accounts:
    "A class representing a group of related accounts."

    # List of accounts, name/value pairs.
    acclist = { }

    # Python supports a single constructor, with the standard name __init__.
    # This constructor creates the first account, and initializes it to a
    # given balance.
    def __init__(self, firstname, initbal):
        if initbal < 0:
            raise ValueError, 'Overdraft on ' + firstname
        self.acclist[firstname] = initbal

    # This method creates a new account, with balance zero.  You can
    # use transfer to put something into it.
    def newacct(self, acctname):
        "Create a new account."
        if self.acclist.has_key(acctname):
            raise KeyError, 'Account name ' + acctname + ' already exists.'
        self.acclist[acctname] = 0

    # Transfer funds from one account to another.  Balances may not
    # become negative.  If an account does not exist, or becomes
    # negative, something will get thrown.
    def transfer(self, fr, to, amt):
        if amt < 0:
            self.transfer(to, fr, -amt)
        else:
            if self.acclist[fr] < amt:
                raise ValueError, 'Overdraft on ' + fr
            else:
                self.acclist[fr] = self.acclist[fr] - amt
                self.acclist[to] = self.acclist[to] + amt

    # This method closes an account.  It will not close an account
    # with a non-zero balance.  If an optional second argument is given, the
    # contents will be transfered to that account before closing.
    def close(self, acctname, receiver = None):
        "Close an account."

        # If there is a receiver, tranfer the funds.
        if receiver:
            self.transfer(acctname, receiver, self.acclist[acctname])

        # Must be empty.
        if self.acclist[acctname]:
            raise ValueError, 'Close of non-empty ' + acctname

        del self.acclist[acctname]

    # Fetch the value of an account.
    def value(self, acctname):
        "Return the value an account."
        return self.acclist[acctname]

    # Get the list of account names.
    def list(self):
        "Get a list of account names."
        return self.acclist.keys()

