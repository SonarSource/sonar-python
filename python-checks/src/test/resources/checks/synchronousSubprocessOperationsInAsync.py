import asyncio
import subprocess
import trio
import anyio
import os


# Asyncio cases
async def asyncio_with_subprocess_run():
    # ^[sc=1;ec=5]> {{This function is async.}}
    result = subprocess.run(["echo", "Hello World"])  # Noncompliant {{Use an async subprocess call in this async function instead of a synchronous one.}}
#            ^^^^^^^^^^^^^^

async def asyncio_with_subprocess_popen():
    proc = subprocess.Popen(["ls", "-l"])  # Noncompliant
    proc.wait()  # Noncompliant

async def asyncio_with_subprocess_call():
    subprocess.call(["date"])  # Noncompliant

async def asyncio_with_subprocess_check_call():
    subprocess.check_call(["whoami"])  # Noncompliant

async def asyncio_with_subprocess_check_output():
    output = subprocess.check_output(["hostname"])  # Noncompliant

async def asyncio_with_os_system():
    os.system("echo Hello")  # Noncompliant

async def asyncio_with_popen_communicate():
    proc = subprocess.Popen(["echo", "test"])  # Noncompliant
    stdout, stderr = proc.communicate()  # Noncompliant

async def asyncio_with_os_spawn():
    os.spawnl(os.P_WAIT, "/bin/echo", "echo", "hello")  # Noncompliant

async def asyncio_with_subprocess_getstatusoutput():
    status, output = subprocess.getstatusoutput("ls")  # Noncompliant

async def asyncio_with_subprocess_getoutput():
    output = subprocess.getoutput("whoami")  # Noncompliant

async def asyncio_compliant():
    proc = await asyncio.create_subprocess_exec(
        "echo", "Hello World",
        stdout=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()

    # Shell version
    shell_proc = await asyncio.create_subprocess_shell(
        "ls -l",
        stdout=asyncio.subprocess.PIPE
    )
    await shell_proc.wait()


# Trio cases
async def trio_with_subprocess_run():
    async with trio.open_nursery() as nursery:
        result = subprocess.run(["ping", "-c", "1", "localhost"])  # Noncompliant

async def trio_with_subprocess_popen():
    async with trio.open_nursery() as nursery:
        proc = subprocess.Popen(["python", "--version"])  # Noncompliant

async def trio_with_popen_communicate():
    async with trio.open_nursery() as nursery:
        proc = subprocess.Popen(["echo", "trio test"])  # Noncompliant
        stdout, stderr = proc.communicate()  # Noncompliant

async def trio_compliant():
    async with trio.open_nursery() as nursery:
        stdout = await trio.run_process(["echo", "Hello Trio"])


# AnyIO cases
async def anyio_with_subprocess_run():
    async with anyio.create_task_group() as tg:
        result = subprocess.run(["find", ".", "-name", "*.py"])  # Noncompliant

async def anyio_with_subprocess_check_output():
    async with anyio.create_task_group() as tg:
        output = subprocess.check_output(["ls", "-la"])  # Noncompliant

async def anyio_with_popen_communicate():
    async with anyio.create_task_group() as tg:
        proc = subprocess.Popen(["cat", "/etc/hostname"])  # Noncompliant
        stdout, stderr = proc.communicate()  # Noncompliant

async def anyio_with_os_spawn():
    async with anyio.create_task_group() as tg:
        os.spawnv(os.P_WAIT, "/bin/ls", ["ls", "-l"])  # Noncompliant

async def anyio_compliant():
    async with anyio.create_task_group() as tg:
        result = await anyio.run_process(["echo", "Hello AnyIO"])

# Test cases outside async functions (should not be flagged)
def sync_function_with_subprocess():
    result = subprocess.run(["echo", "This is fine"])
    proc = subprocess.Popen(["ls"])
    proc.wait()
    proc.communicate()

def sync_function_with_os_calls():
    os.system("echo This is also fine")
    file_obj = os.popen("date")
    os.spawnl(os.P_WAIT, "/bin/echo", "echo", "fine")