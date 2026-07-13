def serialize_user(value):  # Noncompliant {{Refactor this function to reduce its Cognitive Complexity from 8 to the 0 allowed.}}
    if value is None:
        return None
    if value.name == "default":
        if value.email == "default":
            return "default generated value"
        if value.phone == "default":
            return "default generated value"
        if value.address == "default":
            return "default generated value"
