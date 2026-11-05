def direct_item_access(value):
    if isinstance(
        value,
        dict,  # Noncompliant {{Use "Mapping" instead of "dict" here.}}
        # ^[sc=9;ec=12]
    ):
        return value["name"]
    return None


def mapping_method(value):
    if isinstance(value, dict):  # Noncompliant
        # ^[sc=26;ec=29]
        return value.get("name")
    return None


def compliant_mapping_check(value):
    if isinstance(value, Mapping):
        return value["name"]
    return None


def mapping_check_accepts_frozendict():
    value = frozendict(name="Sonar")
    if isinstance(value, Mapping):
        return value["name"]
    return None


def compliant_multiple_statements(value):
    if isinstance(value, dict):
        logger.info(value)
        return value["name"]
    return None


def compliant_mutation(value):
    if isinstance(value, dict):
        value["name"] = "Sonar"
        return value
    return None


def compliant_non_dict_check(value):
    if isinstance(value, list):
        return value[0]
    return None


def compliant_without_a_type_check(value):
    if value:
        return value["name"]
    return None


def compliant_returning_the_value(value):
    if isinstance(value, dict):
        return value
    return None


def compliant_returning_a_constant(value):
    if isinstance(value, dict):
        return "name"
    return None


def compliant_mutating_method(value):
    if isinstance(value, dict):
        return value.pop("name")
    return None


def compliant_attribute_access(value):
    if isinstance(value, dict):
        return value.__class__
    return None


def compliant_without_a_return(value):
    if isinstance(value, dict):
        print(value["name"])


def compliant_multiple_return_values(value):
    if isinstance(value, dict):
        return value["first"], value["second"]
    return None


def compliant_non_name_argument(value):
    if isinstance(value["nested"], dict):
        return value["nested"]["name"]
    return None


def mapping_views(value):
    if isinstance(value, dict):
        return value.items(), value.keys(), value.values()
    return None
