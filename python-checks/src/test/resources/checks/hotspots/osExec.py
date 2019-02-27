import os
import subprocess
from subprocess import run, call, check_call, check_output, Popen

# See https://docs.python.org/3/library/subprocess.html
params = ["ls", "-l"]
subprocess.run(params)  # Noncompliant
run(params)  # Noncompliant
subprocess.Popen(params)  # Noncompliant
Popen(params)  # Noncompliant

# Older API
subprocess.call(params)  # Noncompliant
call(params)  # Noncompliant
subprocess.check_call(params)  # Noncompliant
check_call(params)  # Noncompliant
subprocess.check_output(params)  # Noncompliant
check_output(params)  # Noncompliant

# See https://docs.python.org/3/library/os.html
cmd = "ls -l"
os.system(cmd)  # Noncompliant
mode = os.P_WAIT
file = "ls"
path = "/bin/ls"
env = os.environ
os.spawnl(mode, path, *params)  # Noncompliant
os.spawnle(mode, path, *params, env)  # Noncompliant
os.spawnlp(mode, file, *params)  # Noncompliant
os.spawnlpe(mode, file, *params, env)  # Noncompliant
os.spawnv(mode, path, params)  # Noncompliant
os.spawnve(mode, path, params, env)  # Noncompliant
os.spawnvp(mode, file, params)  # Noncompliant
os.spawnvpe(mode, file, params, env)  # Noncompliant
mode = 'r'
(child_stdout) = os.popen(cmd, mode, 1)  # Noncompliant

# print(child_stdout.read())
(_, output) = subprocess.getstatusoutput(cmd)  # Noncompliant
out = subprocess.getoutput(cmd)  # Noncompliant
os.startfile(path)  # Noncompliant
os.execl(path, *params)  # Noncompliant
os.execle(path, *params, env)  # Noncompliant
os.execlp(file, *params)  # Noncompliant
os.execlpe(file, *params, env)  # Noncompliant
os.execv(path, params)  # Noncompliant
os.execve(path, params, env)  # Noncompliant
os.execvp(file, params)  # Noncompliant
os.execvpe(file, params, env)  # Noncompliant
