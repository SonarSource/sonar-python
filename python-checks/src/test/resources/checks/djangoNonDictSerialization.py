from django.http import JsonResponse

response = JsonResponse([1, 2, 3]) # Noncompliant {{Use a dictionary object here, or set the "safe" flag to False.}}
                       #^^^^^^^^^

list_lit = [42]
response = JsonResponse(list_lit) # Noncompliant {{Use a dictionary object here, or set the "safe" flag to False.}}
                       #^^^^^^^^

response = JsonResponse(4) # Noncompliant {{Use a dictionary object here, or set the "safe" flag to False.}}
                       #^

response = JsonResponse({1,2,3,4}) # Noncompliant {{Use a dictionary object here, or set the "safe" flag to False.}}
                       #^^^^^^^^^

response = JsonResponse({1,2,3,4}, safe=True) # Noncompliant {{Use a dictionary object here, or set the "safe" flag to False.}}
                       #^^^^^^^^^

def MyClass():
  pass

 response = JsonResponse(MyClass())


def safeCall():
    return True

response = JsonResponse({1,2,3,4}, safe=safeCall()) # FN - cannot tell if safeCall would return False

def MyDictClass(dict):
  pass

response = JsonResponse(MyDictClass())

my_var = MyDictClass()
response = JsonResponse(my_var)

import django.core.serializers.json.DjangoJSONEncoder

class Success:
  success  = JsonResponse()
  success1 = JsonResponse({})
  success2 = JsonResponse({"a": "b", "c": "d"})
  success3 = JsonResponse(foo())
  success4 = JsonResponse(4, safe=False)
  success5 = JsonResponse([1, 2, 3], DjangoJSONEncoder, False)
  data = {"foo": 42}
  success6 = JsonResponse(data)
  a = data
  b = a
  c = b
  d = c
  e = d
  f = e
  g = f
  reassigned_data = g
  success7 = JsonResponse(reassigned_data)


def unknown_data(maybe_dict):
    return JsonResponse(maybe_dict)

import something
unknown_var = something.config
JsonResponse(unknown_var)
