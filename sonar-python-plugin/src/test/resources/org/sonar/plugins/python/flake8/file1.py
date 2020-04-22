import os
from Crypto import Util
from twisted.python.hashlib import md5

def foo(secret='secret'):
    os.system("echo " + os.environ['HOME'])
    return md5("%s:%s:%s" % (secret, str(random.random()), str( Util.number.long_to_bytes(42)))).hexdigest()


def bar(things):
    for thing in things:
        if thing != 41:
            if thing != 43:
                if thing > 40:
                    if thing < 44:
                        print("42")
