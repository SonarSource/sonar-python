from custom import library as lib

# SingleArgument
lib.RaiseOnArgument(my_arg = {
    "test" : "bad_value" # Noncompliant
})

lib.RaiseOnDictionary(my_arg = # Noncompliant
{
    "test" : "bad_value"
})

lib.RaiseOnDictionaryWithInterval(my_arg = {
    "min" : 5, # Noncompliant
    "max" : 30
})
