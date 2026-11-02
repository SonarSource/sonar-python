def noncompliant_consecutive_appends(base_command, path):
    command = list(base_command)
    command.append("--archive")  # Noncompliant {{Use "list.extend()" instead of consecutive "append()" calls.}}
#   ^^^^^^^^^^^^^^
    command.append("--verbose")
#   ^^^^^^^^^^^^^^< {{This call is part of the same sequence.}}
    command.append(path)
#   ^^^^^^^^^^^^^^< {{This call is part of the same sequence.}}
    return command


def noncompliant_four_calls():
    my_list = []
    my_list.append(1)  # Noncompliant
    my_list.append(2)
    my_list.append(3)
    my_list.append(4)


def noncompliant_complex_arguments(items, flag):
    my_list = []
    my_list.append(items[0])  # Noncompliant
    my_list.append(items[1] if flag else None)
    my_list.append(len(items))


def noncompliant_after_other_statements():
    my_list = []
    print("start")
    my_list.append(1)  # Noncompliant
    my_list.append(2)
    my_list.append(3)


def noncompliant_two_runs():
    my_list = []
    my_list.append(1)  # Noncompliant
    my_list.append(2)
    my_list.append(3)
    print("separator")
    my_list.append(4)  # Noncompliant
    my_list.append(5)
    my_list.append(6)


def noncompliant_run_after_another_list():
    first = []
    second = []
    first.append(1)
    first.append(2)
    second.append(1)  # Noncompliant
    second.append(2)
    second.append(3)


def noncompliant_nested_block(flag):
    my_list = []
    if flag:
        my_list.append(1)  # Noncompliant
        my_list.append(2)
        my_list.append(3)


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


def compliant_for_loop(other_collection):
    # For-loops are covered by S9409
    my_list = []
    for element in other_collection:
        my_list.append(element)


def compliant_two_consecutive_appends():
    my_list = []
    my_list.append(1)
    my_list.append(2)


def compliant_interleaved_with_other_statements():
    my_list = []
    my_list.append(1)
    print("in between")
    my_list.append(2)
    print("in between")
    my_list.append(3)


def compliant_on_different_lists():
    first = []
    second = []
    third = []
    first.append(1)
    second.append(2)
    third.append(3)


def compliant_referencing_the_list():
    # Grouping these calls would change the result: each argument observes the previous append
    my_list = []
    my_list.append(len(my_list))
    my_list.append(len(my_list))
    my_list.append(len(my_list))


def compliant_in_different_blocks(flag):
    my_list = []
    if flag:
        my_list.append(1)
    else:
        my_list.append(2)
    my_list.append(3)


def compliant_on_unknown_receiver(unknown_receiver):
    unknown_receiver.append(1)
    unknown_receiver.append(2)
    unknown_receiver.append(3)


def get_list():
    return []


def compliant_on_transient_receiver():
    get_list().append(1)
    get_list().append(2)
    get_list().append(3)


def compliant_on_attribute_receiver(obj):
    obj.items = []
    obj.items.append(1)
    obj.items.append(2)
    obj.items.append(3)


def compliant_on_a_set():
    my_set = {1}
    my_set.add(1)
    my_set.add(2)
    my_set.add(3)


def compliant_with_wrong_arity():
    my_list = []
    my_list.append()
    my_list.append(1, 2)
    my_list.append(3)


def compliant_with_keyword_argument():
    my_list = []
    my_list.append(item=1)
    my_list.append(item=2)
    my_list.append(item=3)


def compliant_with_starred_argument(args):
    my_list = []
    my_list.append(*args)
    my_list.append(*args)
    my_list.append(*args)


def compliant_assigned_result():
    my_list = []
    a = my_list.append(1)
    b = my_list.append(2)
    c = my_list.append(3)


def compliant_tuple_expression_statement():
    my_list = []
    my_list.append(1), my_list.append(2)
    my_list.append(3), my_list.append(4)
    my_list.append(5), my_list.append(6)
