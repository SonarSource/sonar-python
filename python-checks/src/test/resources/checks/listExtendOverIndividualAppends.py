
def noncompliant_for_loop():
    my_list = [1, 2, 3]
    other_collection = [4, 5, 6]
    for element in other_collection:
        my_list.append(element)  # Noncompliant {{Use "list.extend()" instead of a for-loop with "append()".}}
    #   ^^^^^^^^^^^^^^


def noncompliant_for_loop_multiline_iterable():
    my_list = []
    for element in [
        1,
        2,
        3,
    ]:
        my_list.append(element)  # Noncompliant
    #   ^^^^^^^^^^^^^^


def noncompliant_for_loop_multiline_receiver():
    my_list = []
    items = [1, 2, 3]
    for item in items:
        (my_list  # Noncompliant
         ).append(item)


def noncompliant_consecutive_appends(base_command, path):
    command = list(base_command)
    command.append("--archive")  # Noncompliant {{Use "list.extend()" instead of consecutive "append()" calls.}}
#   ^^^^^^^^^^^^^^
    command.append("--verbose")
#   ^^^^^^^^^^^^^^< {{This call is part of the same sequence.}}
    command.append(path)
#   ^^^^^^^^^^^^^^< {{This call is part of the same sequence.}}
    return command


def noncompliant_consecutive_appends_four_calls():
    my_list = []
    my_list.append(1)  # Noncompliant
    my_list.append(2)
    my_list.append(3)
    my_list.append(4)


def noncompliant_consecutive_appends_complex_arguments(items, flag):
    my_list = []
    my_list.append(items[0])  # Noncompliant
    my_list.append(items[1] if flag else None)
    my_list.append(len(items))


def noncompliant_consecutive_appends_after_other_statements():
    my_list = []
    print("start")
    my_list.append(1)  # Noncompliant
    my_list.append(2)
    my_list.append(3)


def noncompliant_consecutive_appends_two_runs(other_list):
    my_list = []
    my_list.append(1)  # Noncompliant
    my_list.append(2)
    my_list.append(3)
    print("separator")
    my_list.append(4)  # Noncompliant
    my_list.append(5)
    my_list.append(6)


class MyClass:
    def build(self, path):
        my_list = []
        my_list.append("first")  # Noncompliant
        my_list.append("second")
        my_list.append(path)
        return my_list


def compliant_extend_call():
    my_list = [1, 2, 3]
    other_collection = [4, 5, 6]
    my_list.extend(other_collection)
    my_list.extend(["a", "b", "c"])


def compliant_two_consecutive_appends():
    my_list = []
    my_list.append(1)
    my_list.append(2)


def compliant_appends_interleaved_with_other_statements():
    my_list = []
    my_list.append(1)
    print("in between")
    my_list.append(2)
    print("in between")
    my_list.append(3)


def compliant_appends_on_different_lists():
    first = []
    second = []
    third = []
    first.append(1)
    second.append(2)
    third.append(3)


def compliant_appends_referencing_the_list(items):
    # Grouping these calls would change the result: each argument observes the previous append
    my_list = []
    my_list.append(len(my_list))
    my_list.append(len(my_list))
    my_list.append(len(my_list))


def compliant_appends_in_different_blocks(flag):
    my_list = []
    if flag:
        my_list.append(1)
    else:
        my_list.append(2)
    my_list.append(3)


def compliant_appends_on_unknown_receiver(unknown_receiver):
    unknown_receiver.append(1)
    unknown_receiver.append(2)
    unknown_receiver.append(3)


def get_list():
    return []


def compliant_appends_on_transient_receiver():
    get_list().append(1)
    get_list().append(2)
    get_list().append(3)


def compliant_appends_on_a_set():
    my_set = {1}
    my_set.add(1)
    my_set.add(2)
    my_set.add(3)


def compliant_appends_with_wrong_arity():
    my_list = []
    my_list.append()
    my_list.append(1, 2)
    my_list.append(3)


def compliant_appends_with_keyword_argument():
    my_list = []
    my_list.append(item=1)
    my_list.append(item=2)
    my_list.append(item=3)


def compliant_appends_with_starred_argument(args):
    my_list = []
    my_list.append(*args)
    my_list.append(*args)
    my_list.append(*args)


def compliant_for_loop_tuple_unpacking_loop_var():
    my_list = []
    pairs = [(1, "a"), (2, "b")]
    for k, v in pairs:
        my_list.append(k)


def compliant_for_loop_multiple_iterables():
    my_list = []
    a = [1, 2]
    b = [3, 4]
    for x in a, b:
        my_list.append(x)


def compliant_for_loop_multiple_body_statements():
    my_list = []
    log = []
    items = [1, 2, 3]
    for item in items:
        log.append(item)
        my_list.append(item)


def compliant_for_loop_conditional_in_body():
    my_list = []
    items = range(10)
    for item in items:
        if item % 2 == 0:
            my_list.append(item)


def compliant_for_loop_non_call_in_body():
    my_list = []
    items = [1, 2, 3]
    for item in items:
        item


def compliant_for_loop_expression_tuple_in_body():
    my_list = []
    items = [1, 2, 3]
    for item in items:
        item, item


def compliant_for_loop_bare_function_call():
    my_list = []
    items = [1, 2, 3]
    for item in items:
        print(item)


def compliant_for_loop_method_not_append():
    my_list = []
    items = [1, 2, 3]
    for item in items:
        my_list.remove(item)


def compliant_for_loop_receiver_is_call():
    items = [1, 2, 3]
    for item in items:
        get_list().append(item)


def compliant_for_loop_unknown_receiver_type(unknown_receiver, items):
    for item in items:
        unknown_receiver.append(item)


def compliant_for_loop_no_argument():
    my_list = []
    for item in [1, 2, 3]:
        my_list.append()


def compliant_for_loop_starred_argument():
    my_list = []
    args = (1,)
    for item in [1, 2, 3]:
        my_list.append(*args)


def compliant_for_loop_keyword_argument():
    my_list = []
    for item in [1, 2, 3]:
        my_list.append(item=item)


def compliant_for_loop_transformed_argument():
    my_list = []
    items = [1, 2, 3]
    for item in items:
        my_list.append(item * 2)


def compliant_for_loop_different_variable():
    my_list = []
    other = "fixed_value"
    items = [1, 2, 3]
    for item in items:
        my_list.append(other)


async def compliant_async_for_loop(async_iterable):
    # "extend()" only accepts synchronous iterables
    my_list = []
    async for item in async_iterable:
        my_list.append(item)


def compliant_for_loop_appending_the_iterated_list():
    my_list = [1, 2, 3]
    for item in my_list:
        my_list.append(item)


def compliant_for_else():
    my_list = []
    items = [1, 2, 3]
    for item in items:
        my_list.append(item)
    else:
        fallback()


class MyOtherClass:
    def __init__(self):
        self.seen: list = []

    def process(self, items):
        # FN: instance attribute type not inferred across methods
        for item in items:
            self.seen.append(item)
