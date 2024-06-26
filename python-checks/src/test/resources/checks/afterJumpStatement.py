for i in range(1):
    break

for i in range(1):
    break; print(i) # Noncompliant {{Delete this unreachable code or refactor the code to make it reachable.}}
#          ^^^^^^^^

for i in range(1):
    break
#   ^^^^^> {{Statement exiting the current code block.}}
    print("a") # Noncompliant {{Delete this unreachable code or refactor the code to make it reachable.}}
#   ^[el=+2;ec=14]
    print("b")

for i in range(1):
    continue
    print(i) # Noncompliant

if True:
    print(1)
    raise TypeError("message")
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^> {{Statement exiting the current code block.}}
    if True: pass # Noncompliant 
#      ^^^^
def fun1():
    return 1

def fun2():
    print(1)
    return 2

def fun3():
    return 2
    print(1) # Noncompliant

def fun4():
    return 2;

def fun5():
    return 2; print(1) # Noncompliant

def if_else_return(x):
    if x:
        print(True)
        return
    #   ^^^^^^>
    else:
        print(False)
        return
    #   ^^^^^^>
    print('dead code!') # Noncompliant 
#   ^^^^^^^^^^^^^^^^^^^

def if_else_yield(x):
    if x:
        print(True)
        yield
    else:
        print(False)
        yield
    print('Not dead code!') # OK


def if_else_raise():
    if x:
        print(True)
        raise TypeError("message")
    else:
        print(False)
        raise TypeError("message")
    print('dead code!') # Noncompliant

def while_if_else_break(x):
    while True:
        if x:
            print(True)
            break
        else:
            print(False)
            break
        print('dead code!') # Noncompliant
    print("end of loop")

def while_if_else_continue(x):
    while True:
        if x:
            print(True)
            continue
        else:
            print(False)
            continue
        print('dead code!') # Noncompliant
    print("end of loop")

def for_if_else_break():
    for value in range(1, 10):
        if value > 2:
            print(True)
            break
        else:
            print(False)
            break
        print('dead code!') # Noncompliant
    print("end of loop")

def for_if_else_continue():
    for value in range(1, 10):
        if value > 2:
            print(True)
            continue
        else:
            print(False)
            continue
        print('dead code!') # Noncompliant
    print("end of loop")

def try_stmt_return():
    try:
        print("try")
        return
        print("dead code") # FN
    except Error:
        print("error")

def try_stmt_return_else():
    try:
        print("try")
        return
    except Error:
        print("error")
    else:
        print("dead code") # FN

def try_stmt_return_inside_except():
    try:
        print("try")
        return
    except Error:
        return
        print("error") # FN

def try_stmt_return_except():
    try:
        print("try")
        return
    except Error:
        print("error") # OK

def try_stmt_return_finally():
    try:
        print("try")
        return
    finally:
        print("error") # OK

def with_stmt_raise(expect_raise, e):
    with expect_raise():
        raise e
    print("foo") # OK

# this is semantically incorrect python code, preventing the CFG generation, and so no issue is raised in this case
def return_inside_class():
    class Foo:
        return
    print("foo") # OK

def while_dead_code_in_else_clause(x):
    while x:
        print("foo")
    else:
        raise e
        print("dead code") # Noncompliant

def while_dead_code_in_else_clause_condition_true(x):
    while True:
        if x:
            print(True)
            break
        else:
            print(False)
            break
    else:
        print('dead code!') # FN, we don't take into account while condition being always true
    print("end of loop")

def code_after_with():
    with A():
        return e
    return False


def try_block_having_jump_statement(p):
    while p :
        try:
            if p:
                break
            print("try")
        except AnsibleVaultFormatError as exc:
            raise
        except AnsibleError as e:
            continue
    else:
        raise AnsibleVaultError(msg)
    print("a") # OK

def try_block_without_jump_statements():
    try:
        print("try")
    except E:
        print("except")
    return 42
    print("dead code") # Noncompliant


def match_statement_no_fp(value):
  match value:
    case "1": return
    case "2": return
  print("reachable")


def match_statement_fn(value):
  match value:
    case "1": return
    case x: return
  print("unreachable")  # FN, "case x" will match anything
