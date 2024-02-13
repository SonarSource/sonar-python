print(r"bb\\bb")
print(r"bb\\bb")
print(r"bb\\bb")

def literal_patterns_should_be_excluded(value1, value2, value3):
    match value1:
        case "This is a duplicated pattern":
            print("This is a duplicated literal")

    match value2:
        case "This is a duplicated pattern":
            print("This is a duplicated literal")

    match value3:
        case "This is a duplicated pattern":
            print("This is a duplicated literal")

print("THIS IS A STRING LITERAL")
print("THIS IS A STRING LITERAL")
print("THIS IS A STRING LITERAL")
