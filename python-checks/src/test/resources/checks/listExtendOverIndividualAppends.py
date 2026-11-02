
def noncompliant_for_loop():
    my_list = [1, 2, 3]
    other_collection = [4, 5, 6]
    for element in other_collection:
        my_list.append(element)  # Noncompliant {{Use "list.extend()" to add all elements of "other_collection" instead of a for-loop with "append()".}}
    #   ^^^^^^^^^^^^^^


def noncompliant_for_loop_multiline_iterable():
    my_list = []
    for element in [
        1,
        2,
        3,
    ]:
        my_list.append(element)  # Noncompliant {{Use "list.extend()" to add all elements of the iterated collection instead of a for-loop with "append()".}}
    #   ^^^^^^^^^^^^^^


def noncompliant_for_loop_multiline_receiver():
    my_list = []
    items = [1, 2, 3]
    for item in items:
        (my_list  # Noncompliant
         ).append(item)


def noncompliant_for_loop_named_iterable_message(new_results):
    results = []
    for result in new_results:
        results.append(result)  # Noncompliant {{Use "list.extend()" to add all elements of "new_results" instead of a for-loop with "append()".}}
    #   ^^^^^^^^^^^^^^


def noncompliant_for_loop_parenthesized_iterable_message(new_results):
    results = []
    for result in (new_results):
        results.append(result)  # Noncompliant {{Use "list.extend()" to add all elements of "new_results" instead of a for-loop with "append()".}}
    #   ^^^^^^^^^^^^^^


def noncompliant_for_loop_call_iterable_message():
    my_list = []
    for i in range(10):
        my_list.append(i)  # Noncompliant {{Use "list.extend()" to add all elements of the iterated collection instead of a for-loop with "append()".}}
    #   ^^^^^^^^^^^^^^


def noncompliant_for_loop_attribute_iterable_message(config):
    my_list = []
    for path in config.paths:
        my_list.append(path)  # Noncompliant {{Use "list.extend()" to add all elements of the iterated collection instead of a for-loop with "append()".}}
    #   ^^^^^^^^^^^^^^


def noncompliant_for_loop_subscription_iterable_message(groups):
    my_list = []
    for item in groups[0]:
        my_list.append(item)  # Noncompliant {{Use "list.extend()" to add all elements of the iterated collection instead of a for-loop with "append()".}}
    #   ^^^^^^^^^^^^^^


class MyClass:
    def build(self, items):
        my_list = []
        for item in items:
            my_list.append(item)  # Noncompliant
        return my_list


def compliant_extend_call():
    my_list = [1, 2, 3]
    other_collection = [4, 5, 6]
    my_list.extend(other_collection)
    my_list.extend(["a", "b", "c"])


def compliant_consecutive_appends(base_command, path):
    # Consecutive calls are covered by S9454
    command = list(base_command)
    command.append("--archive")
    command.append("--verbose")
    command.append(path)
    return command


def get_list():
    return []


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
