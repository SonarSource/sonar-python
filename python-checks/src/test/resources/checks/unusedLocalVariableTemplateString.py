def direct_replacement():
    name = "template string"
    return t"{name}"


def conditional_replacement():
    name = "template string"
    enabled = True
    fallback = "fallback"
    return t"{name if enabled else fallback}"


def formatted_string_replacement(name):
    return f"{name}"


def unused_local():
    value = "unused"  # Noncompliant {{Remove the unused local variable "value".}}
