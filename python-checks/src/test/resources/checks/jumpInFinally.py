def continue_statement():
    for i in range(3):
        try:
            foo(i)
        finally:
            continue # Noncompliant {{Remove this "continue" statement from this "finally" block.}}
#           ^^^^^^^^
    print("the end")

def break_statement():
    for i in range(3):
        try:
            foo(i)
        finally:
            break # Noncompliant {{Remove this "break" statement from this "finally" block.}}
    print("the end")

def return_statement():
    try:
        foo(i)
    finally:
        return # Noncompliant {{Remove this "return" statement from this "finally" block.}}
    print("the end")

def not_in_finally():
    for i in range(3):
        continue
    for i in range(3):
        break
    for i in range(3):
        return
    print("the end")

def nested_loop_in_finally():
    for i in range(3):
        try:
            foo(i)
        finally:
            for j in range(3):
                continue
    for i in range(3):
        try:
            foo(i)
        finally:
            for j in range(3):
                break
    for i in range(3):
        try:
            foo(i)
        finally:
            for j in range(3):
                return j # Noncompliant

def jump_in_nested_function():
    for i in range(3):
        try:
            foo(i)
        finally:
            def nested_return():
                return
            def nested_break():
                break
            def nested_continue():
                continue
            nested_return()
            nested_break()
            nested_continue()
    print("the end")

def jump_in_try_or_except():
    for i in range(3):
        try:
            if foo(i):
                break
            else:
                continue
        finally:
            print(42)
    for i in range(3):
        try:
            print(i)
        except Exception as e:
            if foo(e):
                break
            else:
                continue
        finally:
            print(42)

# outside function
if cond:
    try:
        foo(i)
    finally:
        return # Noncompliant
else:
    return
