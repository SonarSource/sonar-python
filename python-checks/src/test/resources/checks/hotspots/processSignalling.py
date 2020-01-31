import os
from os import (kill, killpg)
import my_os

def send_signal(pid, sig, pgid):
    os.kill(pid, sig)  # Noncompliant {{Make sure that sending signals is safe here.}}
#   ^^^^^^^
    os.killpg(pgid, sig)  # Noncompliant

    kill(pid, sig) # Noncompliant {{Make sure that sending signals is safe here.}}
#   ^^^^
    killpg() # Noncompliant

    os.getcwd() # OK
    my_os.kill() # OK
