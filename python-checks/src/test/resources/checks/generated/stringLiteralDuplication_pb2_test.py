def serialize_user(value):
    if value.name == "default":
        return "default generated value"  # Noncompliant {{Define a constant instead of duplicating this literal "default generated value" 3 times.}}
    if value.email == "default":
        return "default generated value"
    return "default generated value"
