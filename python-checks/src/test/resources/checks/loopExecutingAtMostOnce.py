while cond: #Noncompliant
    break;

def foo(cond, cond2):

    while cond:
        print 42

    while cond:
        continue

    while cond: # Noncompliant
#   ^^^^^
        break
#       ^^^^^< {{The loop stops here.}}

    while cond:
        if cond2:
            break

    while cond: # Noncompliant 
#   ^^^^^
        if cond2:
            break
#           ^^^^^<
        else:
            break
#           ^^^^^<

   while cond: # Noncompliant 
#  ^^^^^
       raise error
#      ^^^^^<

def try_statements():

    while cond:
        try:
            return doSomething()
        except:
            print("Try again...")

    while cond: # false negative
        try:
            return doSomething()
        except:
            return 42

    while cond:
        try:
            raise error
        except Error as e:
            print(e)

def invalid_continue():
    continue

def nested_jump_statements(items):
    while True: # Noncompliant 
#   ^^^^^
        for item in items:
            if not item:
                break # should not belong to secondary locations
            print(item)
        break
#       ^^^^^<

def nested_jump_statements_with_else(items, p):
    while p: # Noncompliant 
#   ^^^^^
        for item in items:
            if not item:
                break # should not belong to secondary locations
            print(item)
        break
#       ^^^^^<
    else:
        print("foo")
    print("after")

def nested_jump_statements_with_else_continuing_outer_loop(items, p):
    while p:
        for item in items:
            if not item:
                break
            print(item)
        else:
            print("foo")
            continue
        break
    print("after")

def continue_in_except_block(p):
    i = 0
    while i < 10:
        try:
            i = i + 1
            if p:
                raise
        except E:
            continue
        finally:
            print("finally")
        break # OK
