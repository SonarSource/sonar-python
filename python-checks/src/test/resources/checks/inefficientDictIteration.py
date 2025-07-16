def case1():
    fruit = {'a': 'Apple', 'b': 'Banana'}

    for _, value in fruit.items():  # Noncompliant {{Modify this loop to iterate over the dictionary's values.}}
#                   ^^^^^^^^^^^^^
        ...

    for key, _ in fruit.items():  # Noncompliant {{Modify this loop to iterate directly over the dictionary.}}
#                 ^^^^^^^^^^^^^
        ...

    items1 = fruit.items()
    for _, value in items1:  # Noncompliant
        ...

    items2 = fruit.items()
    for key, _ in items2:  # Noncompliant
        ...

def case2():
    fruit = {'a': 'Apple', 'b': 'Banana'}

    for key, value in fruit.items():
        ...

    for key, _ in fruit.something():
        ...

    items1 = fruit.items()
    something(items1)
    for _, value in items1:
        ...

    for key, _ in fruit.items(), fruit.items():
        ...

    for v, v, v in fruit.items():
        ...

    for k.v, v.v in fruit.items():
        ...