def redundant_return_at_end():
    x = 1
    return #Noncompliant

def non_redundant_return_at_end():
    x = 1
    return x

def non_redundant_return_in_if(cond):
    if cond:
        return
    x = 42

def redundant_return_in_if(cond):
    if cond:
        print("foo")
        return #Noncompliant

def redundant_return_in_if_else(cond):
    if cond:
        x = 10
        return #Noncompliant {{Remove this redundant return.}}
#       ^^^^^^
    else:
        x = 42

def redundant_return_in_if_elif(cond, cond2):
    if cond:
        x = 10
        return #Noncompliant
    elif cond2:
        x = 42

def two_redundant_return_if_else(cond):
    if cond:
        x = 10
        return #Noncompliant
    else:
        x = 42
        return #Noncompliant

def two_non_redundant_return_if_else(cond):
    if cond:
        x = 10
        return
    else:
        x = 42
        return
    x = 2

def two_redundant_return_if_elif(cond1, cond2):
    if cond1:
        x = 10
        return #Noncompliant
    elif cond2:
        x = 42
        return #Noncompliant

def two_non_redundant_return_if_elif(cond1, cond2):
    if cond1:
        x = 10
        return
    elif cond2:
        x = 42
        return
    x = 2


def three_redundant_return_if_elif_else(cond1, cond2):
    if cond1:
        x = 10
        return #Noncompliant
    elif cond2:
        x = 42
        return #Noncompliant
    else:
        x = 11
        return #Noncompliant

def three_non_redundant_return_if_elif_else(cond1, cond2):
    if cond1:
        x = 10
        return
    elif cond2:
        x = 42
        return
    else:
        x = 11
        return
    x = 2

def redundant_continue_in_while(cond):
    while cond:
        print("foo")
        continue # Noncompliant {{Remove this redundant continue.}}

def redundant_continue_in_while_if(cond, p):
    while cond:
        if p:
            print("foo")
            continue # Noncompliant

def redundant_continue_in_while_if_else(cond, p):
    while cond:
        if p:
            print("foo")
            continue # Noncompliant
        else:
            x = 42

def non_redundant_continue_in_while(cond, p):
    while cond:
        if p:
            continue
        x = 42

def redundant_continue_in_nested_while(cond1, cond2):
    while cond1:
        while cond2:
          print("foo")
          continue # Noncompliant [[secondary=-2]]

def non_redundant_continue_in_nested_while(cond1, cond2, p):
    while cond1:
        while cond2:
            if p:
                continue
            x = 42

def redundant_continue_in_for(collection):
    for elem in collection:
        print(elem)
        continue # Noncompliant

def redundant_continue_in_for_if(collection, p):
    for elem in collection:
        if p:
            print("foo")
            continue # Noncompliant

def redundant_continue_in_for_if_else(collection, p):
    for elem in collection:
        if p:
            print("foo")
            continue # Noncompliant
        else:
            x = 42

def non_redundant_continue_in_for(collection, p):
    for elem in collection:
        if p:
            continue
        x = 42

def redundant_continue_in_nested_for(collection1, collection2):
    for elem1 in collection1:
        for elem2 in collection2:
          print(elem2)
          continue # Noncompliant

def non_redundant_continue_in_nested_for(collection1, collection2, p):
    for elem1 in collection1:
        for elem2 in collection2:
            if p:
                continue
            x = 42

# outside function

for elem in collection:
    print(elem)
    continue # Noncompliant

for elem in collection:
    if elem == 1:
      continue
    x = 42

def redundant_return_inside_try_block():
    try:
        print('foo')
        return # FN
    except E as e:
        print(e)

def non_redundant_return_inside_try_block():
    try:
        return # OK
    except E as e:
        print(e)
    else:
        print(42)

def redundant_return_inside_catch_block():
    try:
        pass
    except E as e:
        print(e)
        return # FN
    else:
        print(42)

def redundant_return_inside_catch_block_multiple():
    try:
        pass
    except E as e:
        print(e)
        return # FN
    except E1 as e:
        print(e)
    else:
        print(42)


def redundant_return_inside_catch_block_with_finally():
    try:
        pass
    except E as e:
        print(e)
        return # FN
    finally:
        print(42)

def redundant_return_inside_finally_block():
    try:
        pass
    except E as e:
        print(e)
    else:
        print(42)
    finally:
        print(e)
        return # FN

def non_redundant_return_inside_else_block():
    try:
        pass
    except E as e:
        print(e)
    else:
        print(42)
        return # OK - finally to be executed
    finally:
        print(e)
    print("foo")

def redundant_return_inside_else_block():
    try:
        pass
    except E as e:
        print(e)
    else:
        print(42)
        return # FN

def non_redundant_return_inside_except():
    try:
        pass
    except E as e:
        print(e)
        return # OK
    finally:
        print("finally")
    print("after try")

def non_redundant_return_inside_else():
    try:
        pass
    except E as e:
        print(e)
    else:
        print("else")
        return # OK
    finally:
        print("finally")
    print("after try")

def non_redundant_continue_inside_except(cond):
    while cond:
        try:
            pass
        except E as e:
            print(e)
            continue # OK
        finally:
            print("finally")
        print("after try")

def non_redundant_continue_inside_else(cond):
    while cond:
        try:
            pass
        except E as e:
            print(e)
        else:
            print("else")
            continue # OK
        finally:
            print("finally")
        print("after try")

def redundant_continue_inside_while_stmt_in_except(cond):
    try:
        pass
    except E:
        while cond:
            print("foo")
            continue # FN

def raise_statement():
    raise Error()

def invalid_continue():
    continue


with A() as a:
    while cond:
        print("foo")
        continue # Noncompliant

class Foo:
    while cond:
        print("foo")
        continue # Noncompliant

def single_return():
    return # OK
