
def process_template(t):
    pass

def test_basic_violations():
    template = t"Hello {name}"
    print(template)  # Noncompliant {{This template string should be processed before use.}}
    #     ^^^^^^^^
    output = str(template)  # Noncompliant 

    some_other_value = 2
    print(process_template(template))  # Compliant
    output = str(process_template(template))  # Compliant 
    output = str(some_other_value)

# String operations

def test_string_operations():
    template = t"Template: {value}"

    result1 = "{}".format(template)  # Noncompliant 
    result2 = " ".join((template, "other"))  # Noncompliant
    #                   ^^^^^^^^
    result3 = " ".join([template, "other"])  # Noncompliant
    result4 = " ".join(template)  # Noncompliant

# Boolean context

def test_violation_if_condition():
    template = t"Check: {value}"
    if template:  # Noncompliant 
        pass

    if process_template(template):  # Compliant
        pass

# Comparison operations

def test_violation_equality_check():
    template = t"Test {value}"
    if template == "Test value":  # Noncompliant 
        pass
    if "Test value" == template:  # Noncompliant 
        pass

    if process_template(template) == "Test value":  # Compliant 
        pass

def test_violation_in_operator():
    template = t"Substring {x}"
    if "Sub" in template:  # Noncompliant 
        pass

    if template in "Substring test":  # Noncompliant 
        pass

    if "Sub" in process_template(template):  # Compliant
        pass


# Logging and output

def test_violation_logging():
    import logging
    template = t"Log message: {msg}"
    logging.info(template)  # Noncompliant 
    logging.debug(template)  # Noncompliant 
    logging.info(process_template(template))  # Compliant

# Ternary expressions

def test_violation_ternary():
    template = t"Value: {x}"
    result = template if True else "default"  # Noncompliant 
    #        ^^^^^^^^
    result = "default" if True else template # Noncompliant 

# Type conversions

def test_violation_type_conversions():
    template1 = t"123"
    template2 = t"3.14"
    template3 = t"True"

    # Various type conversions should all be violations
    number1 = int(template1)  # Noncompliant {{This template string should be processed before use.}}
    number2 = float(template2)  # Noncompliant {{This template string should be processed before use.}}
    value = bool(template3)  # Noncompliant {{This template string should be processed before use.}}
    number = int(process_template(template2))  # Compliant

# Directly used in expressions

def test_violation_direct_print():
    print(t"Direct: {value}")  # Noncompliant 
    print(process_template(t"Direct: {value}"))  # Compliant

