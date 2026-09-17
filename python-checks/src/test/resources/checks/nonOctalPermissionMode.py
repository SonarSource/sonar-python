import os
from os import chmod as os_chmod
from pathlib import Path
import pathlib

HIGH_VALUE = "Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777)."

def noncompliant_os_chmod():
    os.chmod("file.txt", 755)  # Noncompliant {{Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777).}}
    #                    ^^^
    os.chmod("file.txt", mode=644)  # Noncompliant {{Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777).}}
    #                         ^^^
    os.chmod(mode=755, path="file.txt")  # Noncompliant {{Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777).}}
    #             ^^^
    os.lchmod("file.txt", 755)  # Noncompliant {{Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777).}}
    #                     ^^^
    os.fchmod(3, 644)  # Noncompliant {{Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777).}}
    #            ^^^
    os_chmod("file.txt", 755)  # Noncompliant {{Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777).}}
    #                    ^^^

def noncompliant_other_os_mode_apis():
    os.mkdir("dir", 755)  # Noncompliant {{Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777).}}
    #               ^^^
    os.makedirs("dir", mode=755)  # Noncompliant {{Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777).}}
    #                       ^^^
    os.open("file.txt", os.O_RDONLY, 644)  # Noncompliant {{Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777).}}
    #                                ^^^
    os.umask(755)  # Noncompliant {{Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777).}}
    #        ^^^
    os.mkfifo("fifo", 644)  # Noncompliant {{Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777).}}
    #                 ^^^
    os.mknod("node", 644)  # Noncompliant {{Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777).}}
    #                ^^^

def noncompliant_path_methods():
    Path("config.ini").chmod(644)  # Noncompliant {{Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777).}}
    #                        ^^^
    Path("config.ini").chmod(mode=755)  # Noncompliant {{Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777).}}
    #                             ^^^
    p = Path("config.ini")
    p.chmod(644)  # Noncompliant {{Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777).}}
    #       ^^^
    pathlib.Path("x").chmod(600)  # Noncompliant {{Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777).}}
    #                       ^^^
    Path("dir").mkdir(mode=755)  # Noncompliant {{Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777).}}
    #                      ^^^
    Path("file").touch(mode=644)  # Noncompliant {{Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777).}}
    #                       ^^^
    Path("link").lchmod(644)  # Noncompliant {{Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777).}}
    #                   ^^^

def noncompliant_underscores():
    os.chmod("file.txt", 7_55)  # Noncompliant {{Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777).}}
    #                    ^^^^

def compliant_octal_and_zero():
    os.chmod("file.txt", 0o755)
    os.chmod("file.txt", 0O644)
    os.chmod("file.txt", 0)
    os.lchmod("file.txt", 0o755)
    os.fchmod(3, 0o644)
    os.mkdir("dir", 0o755)
    os.makedirs("dir", mode=0o755)
    os.open("file.txt", os.O_RDONLY, 0o644)
    os.umask(0o755)
    Path("config.ini").chmod(0o644)
    Path("config.ini").chmod(0)
    Path("dir").mkdir(mode=0o755)
    Path("file").touch(mode=0o644)
    p = Path("config.ini")
    p.chmod(0o600)

def compliant_non_literals_and_other_bases():
    mode = 755
    os.chmod("file.txt", mode)
    os.chmod("file.txt", 0x1ed)
    os.chmod("file.txt", 0b101101101)
    os.chmod("file.txt", unknown)
    Path("x").chmod(mode)
    # Intentional decimal equivalents / full st_mode values
    os.chmod("file.txt", 33152)  # Noncompliant {{Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777).}}
    #                    ^^^^^
    os.chmod("file.txt", 384)  # Noncompliant
    #                    ^^^
    os.chmod("file.txt", 99999999999999999999)  # Noncompliant
    #                    ^^^^^^^^^^^^^^^^^^^^

def compliant_unrelated_calls():
    something.chmod(755)
    chmod("file.txt", 755)

def wrapper(chmod):
    chmod("file.txt", 755)
