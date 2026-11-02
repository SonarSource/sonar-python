def noncompliant_consecutive_adds(first, second):
    my_set = set()
    my_set.add(first)  # Noncompliant {{Use "set.update()" instead of consecutive "add()" calls.}}
#   ^^^^^^^^^^
    my_set.add(second)
#   ^^^^^^^^^^< {{This call is part of the same sequence.}}
    my_set.add("third")
#   ^^^^^^^^^^< {{This call is part of the same sequence.}}
    return my_set


def noncompliant_four_calls():
    my_set = set()
    my_set.add(1)  # Noncompliant
    my_set.add(2)
    my_set.add(3)
    my_set.add(4)


def noncompliant_complex_arguments(items, flag):
    my_set = set()
    my_set.add(items[0])  # Noncompliant
    my_set.add(items[1] if flag else None)
    my_set.add(len(items))


def noncompliant_after_other_statements():
    my_set = set()
    print("start")
    my_set.add(1)  # Noncompliant
    my_set.add(2)
    my_set.add(3)


def noncompliant_two_runs():
    my_set = set()
    my_set.add(1)  # Noncompliant
    my_set.add(2)
    my_set.add(3)
    print("separator")
    my_set.add(4)  # Noncompliant
    my_set.add(5)
    my_set.add(6)


def noncompliant_run_after_another_set():
    first = set()
    second = set()
    first.add(1)
    first.add(2)
    second.add(1)  # Noncompliant
    second.add(2)
    second.add(3)


def noncompliant_nested_block(flag):
    my_set = set()
    if flag:
        my_set.add(1)  # Noncompliant
        my_set.add(2)
        my_set.add(3)


class MyClass:
    def build(self, name):
        tags = set()
        tags.add("archived")  # Noncompliant
        tags.add("verified")
        tags.add(name)
        return tags


def compliant_update_call():
    my_set = {1, 2, 3}
    other_collection = [4, 5, 6]
    my_set.update(other_collection)
    my_set.update({"a", "b", "c"})


def compliant_for_loop(other_collection):
    # For-loops are covered by S8502
    my_set = set()
    for element in other_collection:
        my_set.add(element)


def compliant_two_consecutive_adds():
    my_set = set()
    my_set.add(1)
    my_set.add(2)


def compliant_interleaved_with_other_statements():
    my_set = set()
    my_set.add(1)
    print("in between")
    my_set.add(2)
    print("in between")
    my_set.add(3)


def compliant_on_different_sets():
    first = set()
    second = set()
    third = set()
    first.add(1)
    second.add(2)
    third.add(3)


def compliant_referencing_the_set():
    # Grouping these calls would change the result: each argument observes the previous add
    my_set = set()
    my_set.add(len(my_set))
    my_set.add(len(my_set))
    my_set.add(len(my_set))


def compliant_in_different_blocks(flag):
    my_set = set()
    if flag:
        my_set.add(1)
    else:
        my_set.add(2)
    my_set.add(3)


def compliant_on_unknown_receiver(unknown_receiver):
    unknown_receiver.add(1)
    unknown_receiver.add(2)
    unknown_receiver.add(3)


def get_set():
    return set()


def compliant_on_transient_receiver():
    get_set().add(1)
    get_set().add(2)
    get_set().add(3)


def compliant_on_attribute_receiver(obj):
    obj.tags = set()
    obj.tags.add(1)
    obj.tags.add(2)
    obj.tags.add(3)


def compliant_on_a_list():
    my_list = []
    my_list.append(1)
    my_list.append(2)
    my_list.append(3)


def compliant_with_wrong_arity():
    my_set = set()
    my_set.add()
    my_set.add(1, 2)
    my_set.add(3)


def compliant_with_keyword_argument():
    my_set = set()
    my_set.add(elem=1)
    my_set.add(elem=2)
    my_set.add(elem=3)


def compliant_with_starred_argument(args):
    my_set = set()
    my_set.add(*args)
    my_set.add(*args)
    my_set.add(*args)


def compliant_assigned_result():
    my_set = set()
    a = my_set.add(1)
    b = my_set.add(2)
    c = my_set.add(3)


def compliant_tuple_expression_statement():
    my_set = set()
    my_set.add(1), my_set.add(2)
    my_set.add(3), my_set.add(4)
    my_set.add(5), my_set.add(6)
