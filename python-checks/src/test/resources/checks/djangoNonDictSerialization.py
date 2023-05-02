from django.http import JsonResponse

JsonResponse([1, 2, 3]) # Noncompliant {{Use a dictionary object here, or set the "safe" flag to False.}}
            #^^^^^^^^^

list_lit = [42]
JsonResponse(list_lit) # Noncompliant {{Use a dictionary object here, or set the "safe" flag to False.}}
            #^^^^^^^^

JsonResponse(4) # Noncompliant {{Use a dictionary object here, or set the "safe" flag to False.}}
            #^

JsonResponse({1,2,3,4}) # Noncompliant {{Use a dictionary object here, or set the "safe" flag to False.}}
            #^^^^^^^^^

JsonResponse({1,2,3,4}, safe=True) # Noncompliant {{Use a dictionary object here, or set the "safe" flag to False.}}
            #^^^^^^^^^

assignment_exp_list
if(assignment_exp_list:= [1,2,3]):
  print("OK")
JsonResponse(assignment_exp_list) # Noncompliant {{Use a dictionary object here, or set the "safe" flag to False.}}
            #^^^^^^^^^^^^^^^^^^^

dict_comprehension = {k:42 for k in [1,2,3,]}
JsonResponse(dict_comprehension)

def MyClass():
  dict_field = {"a": 42}
  list_field = [1,2,3]

response = JsonResponse(MyClass())
non_dict_obj = MyClass()
JsonResponse(non_dict_obj)
JsonResponse(non_dict_obj.dict_field)
JsonResponse(non_dict_obj.list_field) # FN


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
  data: dict[str, int] = {"foo": 42}
  success6 = JsonResponse(data)
  annotated_only : list[str]
  success7 = JsonResponse(annotated_only)
  a = [1,2,3,4,5]
  b = a
  c = b
  d = c
  e = d
  f = e
  g = f
  success8 = JsonResponse(g)
  assignment_exp = None
  if(assignment_exp:= data):
    print("OK")
  success9 = JsonResponse(assignment_exp)


def unknown_data(maybe_dict):
    return JsonResponse(maybe_dict)

import something
unknown_var = something.config
JsonResponse(unknown_var)
