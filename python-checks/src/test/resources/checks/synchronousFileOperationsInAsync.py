async def open_read_noncompliant():
    # ^[sc=1;ec=5]> {{This function is async.}}
    with open("file.txt", "r") as file:  # Noncompliant {{Use an asynchronous file API instead of synchronous open() in this async function.}}
#        ^^^^^^^^^^^^^^^^^^^^^
        content = file.read()
    return content


async def open_write_noncompliant():
    with open("output.txt", "w") as file:  # Noncompliant
        file.write("data")


async def open_with_mode_noncompliant():
    file = open("data.bin", "rb")  # Noncompliant
    content = file.read()
    file.close()
    return content


async def nested_open_noncompliant():
    def inner_function():
        return open("inner.txt", "r")  # OK - not in async context

    with open("outer.txt", "r") as file:  # Noncompliant
        content = file.read()
    return content


# Compliant examples - asyncio/aiofiles

import aiofiles


async def open_read_compliant_asyncio():
    async with aiofiles.open("file.txt", "r") as file:  # Compliant
        content = await file.read()
    return content


async def open_write_compliant_asyncio():
    async with aiofiles.open("output.txt", "w") as file:  # Compliant
        await file.write("data")


# Compliant examples - trio

import trio


async def open_read_compliant_trio():
    async with await trio.open_file("file.txt", "r") as file:  # Compliant
        content = await file.read()
    return content


async def open_write_compliant_trio():
    async with await trio.open_file("output.txt", "w") as file:  # Compliant
        await file.write("data")


# Compliant examples - anyio

import anyio


async def open_read_compliant_anyio():
    async with await anyio.open_file("file.txt", "r") as file:  # Compliant
        content = await file.read()
    return content


async def open_write_compliant_anyio():
    async with await anyio.open_file("output.txt", "w") as file:  # Compliant
        await file.write("data")


# Edge cases


def sync_function_with_open():
    with open("file.txt", "r") as file:  # OK - not in async function
        return file.read()


async def read_file_other_ways():
    # os.open is lower level and synchronous, could be prone to FP in some cases
    # We could consider excluding this API
    import os

    fd = os.open("file.txt", os.O_RDONLY)  # Noncompliant {{Use an asynchronous file API instead of synchronous os.open() in this async function.}}
    os.close(fd)


# --- pathlib.Path.open tests ---

from pathlib import Path


async def async_pathlib_open_noncompliant():
    p = Path("file.txt")
    with p.open("r") as file:  # Noncompliant {{Use an asynchronous file API instead of synchronous pathlib.Path.open() in this async function.}}
        content = file.read()
    return content


async def async_pathlib_open_write_noncompliant():
    p = Path("output.txt")
    with p.open("w") as file:  # Noncompliant
        file.write("data")


async def async_pathlib_open_import_noncompliant():
    import pathlib

    p = pathlib.Path("data.bin")
    with p.open("rb") as file:  # Noncompliant
        content = file.read()
    return content


def sync_pathlib_open_ok():
    p = Path("file.txt")
    with p.open("r") as file:  # OK - not in async function
        return file.read()


# Compliant example for contrast
async def async_pathlib_open_compliant():
    async with aiofiles.open("file.txt", "r") as file:  # Compliant
        content = await file.read()
    return content


# --- Additional noncompliant cases for missing synchronous file operations ---

import io
import codecs
import os
import tempfile
import gzip
import bz2
import lzma


async def io_open_noncompliant():
    with io.open("file.txt", "r") as file:  # FN because of TypeInferenceV2
        content = file.read()
    return content


async def codecs_open_noncompliant():
    with codecs.open("file.txt", "r", encoding="utf-8") as file:  # Noncompliant {{Use an asynchronous file API instead of synchronous codecs.open() in this async function.}}
        content = file.read()
    return content


async def os_fdopen_noncompliant():
    fd = os.open("file.txt", os.O_RDONLY)  # Noncompliant {{Use an asynchronous file API instead of synchronous os.open() in this async function.}}
    with os.fdopen(fd, "r") as file:  # Noncompliant {{Use an asynchronous file API instead of synchronous os.fdopen() in this async function.}}
        content = file.read()
    return content


async def os_popen_noncompliant():
    with os.popen("cat file.txt", "r") as file:  # Noncompliant {{Use an asynchronous file API instead of synchronous os.popen() in this async function.}}
        content = file.read()
    return content


async def tempfile_temporaryfile_noncompliant():
    with tempfile.TemporaryFile("w+") as file:  # Noncompliant {{Use an asynchronous file API instead of synchronous tempfile.TemporaryFile() in this async function.}}
        file.write("data")
        file.seek(0)
        content = file.read()
    return content


async def tempfile_namedtemporaryfile_noncompliant():
    with tempfile.NamedTemporaryFile("w+") as file:  # Noncompliant {{Use an asynchronous file API instead of synchronous tempfile.NamedTemporaryFile() in this async function.}}
        file.write("data")
        file.seek(0)
        content = file.read()
    return content


async def tempfile_spooledtemporaryfile_noncompliant():
    with tempfile.SpooledTemporaryFile(max_size=10, mode="w+") as file:  # Noncompliant {{Use an asynchronous file API instead of synchronous tempfile.SpooledTemporaryFile() in this async function.}}
        file.write("data")
        file.seek(0)
        content = file.read()
    return content


async def gzip_open_noncompliant():
    with gzip.open("file.txt.gz", "rt") as file:  # Noncompliant {{Use an asynchronous file API instead of synchronous gzip.open() in this async function.}}
        content = file.read()
    return content


async def bz2_open_noncompliant():
    with bz2.open("file.txt.bz2", "rt") as file:  # Noncompliant {{Use an asynchronous file API instead of synchronous bz2.open() in this async function.}}
        content = file.read()
    return content


async def lzma_open_noncompliant():
    with lzma.open("file.txt.xz", "rt") as file:  # Noncompliant {{Use an asynchronous file API instead of synchronous lzma.open() in this async function.}}
        content = file.read()
    return content


# --- Compliant example for aiofiles with gzip ---
async def aiofiles_gzip_compliant():
    # aiofiles does not support gzip directly, so this is just for contrast
    async with aiofiles.open("file.txt", "r") as file:  # Compliant
        content = await file.read()
    return content