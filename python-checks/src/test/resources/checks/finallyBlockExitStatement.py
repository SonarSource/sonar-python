# Test file for FinallyBlockExitStatementCheck

# Basic return in finally block
def return_in_finally():
    try:
        do_something()
    finally:
        cleanup()
        return "done"  # Noncompliant {{Remove this return statement from the finally block.}}

# Compliant: return after try-finally
def return_after_finally():
    result = None
    try:
        do_something()
        result = "done"
    finally:
        cleanup()
    return result  # Compliant

# Break in finally block
def break_in_finally():
    while True:
        try:
            process()
        finally:
            log()
            break  # Noncompliant {{Remove this break statement from the finally block.}}

# Compliant: break after finally
def break_after_finally():
    while True:
        should_break = False
        try:
            process()
            should_break = True
        finally:
            log()
        if should_break:
            break  # Compliant

# Continue in finally block
def continue_in_finally():
    for item in items:
        try:
            process(item)
        finally:
            cleanup()
            continue  # Noncompliant {{Remove this continue statement from the finally block.}}

# Compliant: continue removed from finally
def continue_compliant():
    for item in items:
        try:
            process(item)
        finally:
            cleanup()  # Compliant

# Nested try-finally with return
def nested_try_return():
    try:
        outer_operation()
    finally:
        try:
            inner_operation()
        finally:
            return True  # Noncompliant {{Remove this return statement from the finally block.}}

# Compliant: nested try-finally without return in finally
def nested_try_compliant():
    result = False
    try:
        outer_operation()
    finally:
        try:
            inner_operation()
            result = True
        finally:
            pass
    return result  # Compliant


# Compliant: try-except-finally without return in finally
def with_except_and_finally_compliant():
    result = None
    try:
        risky_op()
    except Exception:
        handle_error()
    finally:
        cleanup()
    return result  # Compliant

# Break in finally with conditional
def loop_with_finally_break():
    while condition():
        try:
            work()
        finally:
            if error:
                break  # Noncompliant {{Remove this break statement from the finally block.}}

# Compliant: break moved after finally
def loop_with_finally_compliant():
    while condition():
        should_break = False
        try:
            work()
        finally:
            if error:
                should_break = True
        if should_break:
            break  # Compliant

# Return inside nested function in finally - should be compliant
def return_in_nested_function():
    try:
        do_something()
    finally:
        def inner_function():
            return "inner"  # Compliant - not exiting finally block
        inner_function()

# Return inside lambda in finally - should be compliant
def return_in_lambda():
    try:
        do_something()
    finally:
        func = lambda: "value"  # Compliant - not exiting finally block
        func()

# Break inside nested loop in finally - should be compliant
def break_in_nested_loop():
    try:
        do_something()
    finally:
        for i in range(10):
            if i == 5:
                break  # Compliant - breaking from loop inside finally, not exiting finally

# Continue inside nested loop in finally - should be compliant
def continue_in_nested_loop():
    try:
        do_something()
    finally:
        for i in range(10):
            if i == 5:
                continue  # Compliant - continuing loop inside finally, not exiting finally

# Multiple returns in finally
def multiple_returns_in_finally():
    try:
        do_something()
    finally:
        if condition1:
            return 1  # Noncompliant {{Remove this return statement from the finally block.}}
        elif condition2:
            return 2  # Noncompliant {{Remove this return statement from the finally block.}}
        else:
            return 3  # Noncompliant {{Remove this return statement from the finally block.}}

# Return in deeply nested finally
def deeply_nested_finally():
    try:
        operation1()
    finally:
        try:
            operation2()
        finally:
            try:
                operation3()
            finally:
                return "deep"  # Noncompliant {{Remove this return statement from the finally block.}}

# Break in for loop containing try-finally
def for_loop_with_break():
    for item in items:
        try:
            process(item)
        finally:
            cleanup()
            break  # Noncompliant {{Remove this break statement from the finally block.}}



# Break in nested for loop inside while
def nested_loops_break():
    while running:
        for item in items:
            try:
                process(item)
            finally:
                break  # Noncompliant {{Remove this break statement from the finally block.}}

# Break in match statement inside finally (Python 3.10+)
def break_in_match_finally():
    while condition:
        try:
            value = get_value()
        finally:
            match value:
                case 1:
                    break  # Noncompliant {{Remove this break statement from the finally block.}}
                case _:
                    pass

def nested_loops_continue():
    for x in range(10):
        while condition(x):
            try:
                work(x)
            finally:
                continue  # Noncompliant {{Remove this continue statement from the finally block.}}

# Return inside class definition in finally - should be compliant
def return_in_class():
    try:
        do_something()
    finally:
        class MyClass:
            def method(self):
                return "value"  # Compliant - not exiting finally block


# Break with label-like comment (Python doesn't have labeled breaks)
def break_with_comment():
    while outer_condition:
        try:
            inner_work()
        finally:
            # Break outer loop
            break  # Noncompliant {{Remove this break statement from the finally block.}}

# Return in finally after exception handling
def return_after_except():
    try:
        risky_operation()
    except ValueError:
        log_error()
    except TypeError:
        log_type_error()
    finally:
        return "completed"  # Noncompliant {{Remove this return statement from the finally block.}}

# Compliant: return in except, not in finally
def return_in_except():
    try:
        operation()
    except Exception:
        return None  # Compliant - in except, not finally
    finally:
        cleanup()

# Compliant: break in try, not in finally
def break_in_try():
    while condition():
        try:
            work()
            if done:
                break  # Compliant - in try, not finally
        finally:
            cleanup()

# Compliant: generator with yield in finally (not return)
def generator_yield_finally():
    try:
        do_work()
    finally:
        yield "cleanup"  # Compliant - yield, not return/break/continue

