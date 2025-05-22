empty_dict1 = dict()  # Noncompliant {{Replace this constructor call with a literal.}}
#             ^^^^
empty_list = list()  # Noncompliant
empty_tuple = tuple()  # Noncompliant
empty_set = set()

user = dict(name="John", age=30)  # Noncompliant

empty_dict2 = {}
empty_list2 = []
empty_tuple2 = ()

user2 = {"name": "John", "age": 30}

nums_list = list([1, 2, 3])
nums_tuple = tuple([1, 2, 3])
combined_dict = dict({"a": 1})


dict_from_items = dict([('a', 1), ('b', 2)])
dict_from_mapping = dict({'one': 1, 'two': 2})

def passing_variable(collection):
    list(collection)
    tuple(collection)
    set(collection)
    dict(collection)
    dict(collection, b=2)
    dict(*collection)
    dict(**collection)

dict_comp = dict((k, v) for k, v in [('a', 1), ('b', 2)])
list_comp = list(x for x in range(3))
tuple_comp = tuple(x for x in range(3))

dict({"a": 1}, b=2)

some_unrelated_method()
a.some_unrelated_method()

def overwriting_collection_constructors():
    def list(): pass
    def tuple(): pass
    def set(): pass
    def dict(): pass

    # compliant since the built-in functions are shadowed
    list()
    tuple()
    set()
    dict()
