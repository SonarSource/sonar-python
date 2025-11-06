from string.templatelib import Interpolation 

# Basic non-compliant case with if/elif isinstance checks
def process_template(template):
    result = []
    for item in template:
        if isinstance(item, str):  # Noncompliant {{Use structural pattern matching (match/case) instead of isinstance() checks for template string processing.}}
        #  ^^^^^^^^^^
            result.append(item.lower())
        elif isinstance(item, Interpolation):
        #    ^^^^^^^^^^< {{Replace this isinstance with the appropriate pattern matching case.}}
            result.append(str(item.value).upper())
    return "".join(result)


# Non-compliant with reversed order of checks
def reverse_check_order(parts):
    result = []
    for part in parts:
        if isinstance(part, Interpolation):  # Noncompliant
            result.append(f"interp: {part.value}")
        elif isinstance(part, str):
            result.append(f"str: {part}")
    return result


# Non-compliant with additional elif branches
def multi_branch_check(elements):
    output = []
    for elem in elements:
        if isinstance(elem, str):  # Noncompliant
        #  ^^^^^^^^^^
            output.append(elem)
        elif isinstance(elem, Interpolation):
        #    ^^^^^^^^^^<
            output.append(str(elem.value))
        elif isinstance(elem, int):
        #    ^^^^^^^^^^<
            output.append(str(elem))
    return output


# Non-compliant with while loop
def while_loop_processing(template):
    result = []
    i = 0
    while i < len(template):
        item = template[i]
        if isinstance(item, str):  # FN not really a strong use case
            result.append(item)
        elif isinstance(item, Interpolation):
            result.append(str(item.value))
        i += 1
    return result


# Non-compliant with enumerate
def enumerate_processing(template):
    result = []
    for idx, item in enumerate(template): 
        if isinstance(item, str): # Noncompliant
            result.append(f"{idx}: {item}")
        elif isinstance(item, Interpolation):
            result.append(f"{idx}: {item.value}")
    return result


# Non-compliant with zip
def zip_processing(template1, template2):
    result = []
    for item1, item2 in zip(template1, template2):  
        if isinstance(item1, str): # Noncompliant
            result.append(item1)
        elif isinstance(item1, Interpolation):
            result.append(str(item1.value))
    return result


# Non-compliant with only if/else (not elif)
def if_else_processing(parts):
    output = []
    for part in parts: 
        if isinstance(part, str): # Noncompliant
            output.append(part.lower())
        else:
            if isinstance(part, Interpolation):
                output.append(str(part.value).upper())
    return output


def generator_body_processing(template):
    result = []
    for item in (x for x in template):  
        if isinstance(item, str): # Noncompliant
            result.append(item)
        elif isinstance(item, Interpolation):
            result.append(str(item.value))
    return result


def comprehension_unpack(template):
    result = []
    for item in [x for x in template]:  
        if isinstance(item, str): # Noncompliant
            result.append(item.title())
        elif isinstance(item, Interpolation):
            result.append(repr(item.value))
    return result


def multiple_isinstance_same_if(items):
    result = []
    for item in items:
        if isinstance(item, str) or isinstance(item, bytes):  # Noncompliant
            result.append(str(item))
        elif isinstance(item, Interpolation):
            result.append(str(item.value))
    return result


def nested_loops(templates):
    result = []
    for template in templates:
        for item in template: 
            if isinstance(item, str):  # Noncompliant
                result.append(item)
            elif isinstance(item, Interpolation):
                result.append(str(item.value))
    return result


def no_isinstance(template):
    result = []
    for item in template:  # Compliant
        result.append(str(item))
    return result


def different_types(items):
    result = []
    for item in items:  # Compliant - not template-related types
        if isinstance(item, list):
            result.extend(item)
        elif isinstance(item, dict):
            result.append(str(item))
    return result


def format_template_compliant(template):
    formatted = []
    for element in template:  # Compliant
        match element:
            case str() as s:
                formatted.append(s.strip())
            case Interpolation() as i:
                formatted.append(f"{i.value}")
            case _:
                formatted.append(str(element))
    return formatted


def regular_isinstance_check(items):
    result = []
    for item in items:  # Compliant - not template processing
        if isinstance(item, int):
            result.append(item * 2)
        elif isinstance(item, float):
            result.append(item * 3)
    return result


def single_isinstance(parts):
    result = []
    for part in parts:  # Compliant - only one isinstance check
        if isinstance(part, str):
            result.append(part.upper())
    return result


def no_iteration(item):
    if isinstance(item, str):  # Compliant - not in loop
        return item.lower()
    elif isinstance(item, Interpolation):
        return str(item.value)
    return None

def other_var(items):
    for t in items:
        other = foo()
        if isinstance(other, str):
            pass
        elif isinstance(other, Interpolation):
            pass

def direct_operations(parts):
    return [p.upper() for p in parts if hasattr(p, "upper")]  # Compliant


def using_type_check(parts):
    result = []
    for part in parts:
        if type(part) == str:
            result.append(part)
        elif type(part) == Interpolation:
            result.append(str(part.value))
    return result

# Compliant - nested loops with match
def nested_loops_compliant(templates):
    result = []
    for template in templates:
        for item in template:  # Compliant
            match item:
                case str():
                    result.append(item)
                case Interpolation():
                    result.append(str(item.value))
    return result

# --------- COVERAGE -----------

def foo():
    pass

def incorrect_isinstance(templates):
    for t in templates:
        if isinstance(t):
            pass
        elif isinstance(foo(), str):
            pass
        elif isinstance(t, foo()):
            pass

