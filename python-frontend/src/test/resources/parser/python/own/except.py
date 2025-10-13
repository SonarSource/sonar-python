try:
    foo()
except FileNotFoundError as e:
    print(e)
except (ChildProcessError, EOFError):
    print("Tuple syntax")
except ArithmeticError, ValueError:
    print("New python 3.14 syntax")
except (BrokenPipeError, BufferError) as e:
    print("Parenthesis are required")
except (MemoryError, OverflowError), TypeError:
    # we still manage to parse it even if it is not needed
    print("Rejected PEP-758 idea - mixed Parenthesis and unparenthesis")
# except AssertionError, AttributeError as e:
#     # parsing will fail for this clause
#     print("Rejected PEP-758 idea - no parenthesis with as")
except OSError, e:
    print("Old python 2 syntax")
