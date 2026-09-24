def relay(values):
    for value in values:  # Noncompliant {{Replace this loop with a "yield from" statement.}}
#   ^^^
        yield value
#       ^^^^^^^^^^^< {{This "yield" only relays the loop variable.}}


def read_files(paths):
    for path in paths:
        with open(path) as file:
            for line in file:  # Noncompliant
                yield line


def nested_loops(matrix):
    for row in matrix:
        for cell in row:  # Noncompliant
            yield cell


def relay_call_result():
    for value in compute():  # Noncompliant
        yield value


class Deck:
    def __iter__(self):
        for card in self._cards:  # Noncompliant
            yield card


def already_delegating(values):
    yield from values


def delegates_inside_loop(iterables):
    for iterable in iterables:
        yield from iterable


def transforms_the_value(values):
    for value in values:
        yield value * 2


def yields_an_attribute(values):
    for value in values:
        yield value.name


def yields_another_name(values, other):
    for value in values:
        yield other


def yields_a_tuple_element(pairs):
    for key, value in pairs:
        yield key


def yields_the_whole_target(pairs):
    for key, value in pairs:
        yield key, value


def yields_a_one_element_tuple(items):
    for value in items:
        yield value,


def unpacks_a_one_element_tuple(rows):
    for value, in rows:
        yield value


def does_more_than_yield(values):
    for value in values:
        print(value)
        yield value


def yields_twice(values):
    for value in values:
        yield value
        yield value


def has_an_else_clause(values):
    for value in values:
        yield value
    else:
        print("done")


def yields_nothing(values):
    for value in values:
        yield


def uses_the_yield_result(values):
    for value in values:
        received = yield value


def uses_the_loop_variable_after_the_loop(values):
    for value in values:
        yield value
    print(value)


def sibling_loops_reuse_the_name(first, second):
    for value in first:  # Noncompliant
        yield value
    for value in second:  # Noncompliant
        yield value


def loop_variable_is_bound_before_the_loop(values):
    value = None
    for value in values:  # Noncompliant
        yield value


def loop_variable_is_rebound_before_being_read(values):
    for value in values:  # Noncompliant
        yield value
    value = "done"
    print(value)


def rebinding_reads_the_loop_variable(values):
    for value in values:
        yield value
    value = transform(value)
    yield value


def annotation_after_the_loop_binds_nothing(values):
    for value in values:
        yield value
    value: int
    print(value)


def annotated_assignment_after_the_loop_rebinds(values):
    for value in values:  # Noncompliant
        yield value
    value: int = 3
    print(value)


def loop_variable_is_compound_assigned_after_the_loop(values):
    value = 0
    for value in values:
        yield value
    value += 1
    print(value)


def loop_variable_is_rebound_by_a_with_statement(values, path):
    for value in values:  # Noncompliant
        yield value
    with open(path) as value:
        print(value)


def does_not_yield(values):
    for value in values:
        print(value)


def returns_instead_of_yielding(values):
    for value in values:
        return value


async def async_generator(values):
    async for value in values:
        yield value


async def async_function_with_a_plain_loop(values):
    for value in values:
        yield value


def nested_function_is_the_generator(values):
    async def inner():
        for value in values:
            yield value

    return inner
