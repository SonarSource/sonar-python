def noncompliant_items_and_attributes(data: frozendict[str, str]):
    # Noncompliant@+1 {{Remove this assignment to an item or attribute of a frozendict.}}
    data["key"] = "value"
    # ^[sc=5;ec=8]
    # Noncompliant@+1 {{Remove this assignment to an item or attribute of a frozendict.}}
    data.attribute = "value"
    # ^[sc=5;ec=8]
    # Noncompliant@+1 {{Remove this assignment to an item or attribute of a frozendict.}}
    data["key"] += "value"
    # ^[sc=5;ec=8]
    # Noncompliant@+1 {{Remove this assignment to an item or attribute of a frozendict.}}
    data["key"]: str = "value"
    # ^[sc=5;ec=8]
    # Noncompliant@+1 {{Remove this assignment to an item or attribute of a frozendict.}}
    (data["key"], other) = "value", "other"
    # ^[sc=6;ec=9]
    # Noncompliant@+1 {{Remove this assignment to an item or attribute of a frozendict.}}
    [data["key"], other] = ["value", "other"]
    # ^[sc=6;ec=9]


def compliant_reads_and_new_values(data: frozendict[str, str]):
    value = data["key"]
    updated_data = frozendict({**data, "key": "value"})
    return value, updated_data


def compliant_mutable_mapping(data: dict[str, str]):
    data["key"] = "value"
    data.attribute = "value"
