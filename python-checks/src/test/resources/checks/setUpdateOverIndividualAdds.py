
def noncompliant_basic():
    my_set = {1, 2, 3}
    other_collection = [4, 5, 6]
    for element in other_collection:
        my_set.add(element)  # Noncompliant {{Use "set.update()" instead of a for-loop with "add()".}}
    #   ^^^^^^^^^^


def noncompliant_multiline_iterable():
    my_set = set()
    for element in [
        1,
        2,
        3,
    ]:
        my_set.add(element)  # Noncompliant
    #   ^^^^^^^^^^


def noncompliant_multiline_receiver():
    my_set = set()
    items = [1, 2, 3]
    for item in items:
        (my_set  # Noncompliant
         ).add(item)


def noncompliant_consecutive_adds(first, second):
    my_set = set()
    my_set.add(first)  # Noncompliant {{Use "set.update()" instead of consecutive "add()" calls.}}
#   ^^^^^^^^^^
    my_set.add(second)
#   ^^^^^^^^^^< {{This call is part of the same sequence.}}
    my_set.add("third")
#   ^^^^^^^^^^< {{This call is part of the same sequence.}}
    return my_set


def noncompliant_consecutive_adds_after_other_statements():
    my_set = set()
    print("start")
    my_set.add(1)  # Noncompliant
    my_set.add(2)
    my_set.add(3)


def compliant_two_consecutive_adds():
    my_set = set()
    my_set.add(1)
    my_set.add(2)


def compliant_adds_interleaved_with_other_statements():
    my_set = set()
    my_set.add(1)
    print("in between")
    my_set.add(2)
    print("in between")
    my_set.add(3)


def compliant_adds_on_different_sets():
    first = set()
    second = set()
    third = set()
    first.add(1)
    second.add(2)
    third.add(3)


def compliant_adds_referencing_the_set():
    # Grouping these calls would change the result: each argument observes the previous add
    my_set = set()
    my_set.add(len(my_set))
    my_set.add(len(my_set))
    my_set.add(len(my_set))


async def compliant_async_for_loop(async_iterable):
    # "update()" only accepts synchronous iterables
    my_set = set()
    async for item in async_iterable:
        my_set.add(item)


def compliant_loop_adding_the_iterated_set():
    my_set = {1, 2, 3}
    for item in my_set:
        my_set.add(item)


def compliant_tuple_unpacking_loop_var():
    my_set = set()
    pairs = [(1, "a"), (2, "b")]
    for k, v in pairs:
        my_set.add(k)


def compliant_tuple_pattern_loop_var():
    my_set = set()
    pairs = [(1, 2), (3, 4)]
    for (a, b) in pairs:
        my_set.add(a)


def compliant_multiple_iterables():
    my_set = set()
    a = [1, 2]
    b = [3, 4]
    for x in a, b:
        my_set.add(x)


def compliant_multiple_body_statements():
    my_set = set()
    log = []
    items = [1, 2, 3]
    for item in items:
        log.append(item)
        my_set.add(item)


def compliant_conditional_in_body():
    my_set = set()
    items = range(10)
    for item in items:
        if item % 2 == 0:
            my_set.add(item)


def compliant_expression_tuple_in_body():
    my_set = set()
    items = [1, 2, 3]
    for item in items:
        item, item


def compliant_non_call_in_body():
    my_set = set()
    items = [1, 2, 3]
    for item in items:
        item


def compliant_bare_function_call():
    my_set = set()
    items = [1, 2, 3]
    for item in items:
        print(item)


def compliant_method_not_add():
    my_set = set()
    items = [1, 2, 3]
    for item in items:
        my_set.discard(item)


def get_set():
    return set()


def compliant_receiver_is_call():
    items = [1, 2, 3]
    for item in items:
        get_set().add(item)


def compliant_unknown_receiver_type(unknown_receiver, items):
    for item in items:
        unknown_receiver.add(item)


def compliant_no_argument():
    my_set = set()
    for item in [1, 2, 3]:
        my_set.add()


def compliant_starred_argument():
    my_set = set()
    args = (1,)
    for item in [1, 2, 3]:
        my_set.add(*args)


def compliant_keyword_argument():
    my_set = set()
    for item in [1, 2, 3]:
        my_set.add(elem=item)


def compliant_transformed_argument():
    my_set = set()
    items = [1, 2, 3]
    for item in items:
        my_set.add(item * 2)


def compliant_different_variable():
    my_set = set()
    other = "fixed_value"
    items = [1, 2, 3]
    for item in items:
        my_set.add(other)


def compliant_for_else():
    my_set = set()
    items = [1, 2, 3]
    for item in items:
        my_set.add(item)
    else:
        fallback()


class MyClass:
    def __init__(self):
        self.seen: set = set()

    def process(self, items):
        # FN: instance attribute type not inferred across methods
        for item in items:
            self.seen.add(item)
