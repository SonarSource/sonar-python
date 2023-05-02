from django.shortcuts import render

def using_locals_from_var(request):
    username = "alice"
    password = "p@ssw0rd"
    context = locals()
             #^^^^^^^^> {{locals() is assigned to "context" here.}}
    return render(request, "my_template.html", context) # Noncompliant {{Use an explicit context instead of passing "locals()" to this Django "render" call.}}
                                              #^^^^^^^
def using_locals_from_var(request):
    class MyContextClass():
      attr_ref : Any

    username = "alice"
    password = "p@ssw0rd"
    obj = MyContextClass()
    obj.attr_ref : Any = locals()
    return render(request, "my_template.html", obj.attr_ref) # FN - not supporting object attributes

def using_locals_with_named_params(request):
    username = "alice"
    password = "p@ssw0rd"
    context = locals()
             #^^^^^^^^> {{locals() is assigned to "context" here.}}
    return render(request, "my_template.html", content_type=None, context=context) # Noncompliant {{Use an explicit context instead of passing "locals()" to this Django "render" call.}}
                                                                         #^^^^^^^
def using_annotated_locals(request):
    username = "alice"
    password = "p@ssw0rd"
    context: Any = locals()
                  #^^^^^^^^> {{locals() is assigned to "context" here.}}
    return render(request, "my_template.html", content_type=None, context=context) # Noncompliant {{Use an explicit context instead of passing "locals()" to this Django "render" call.}}
                                                                         #^^^^^^^

def using_assignment_expression(request):
    username = "alice"
    password = "p@ssw0rd"
    if(context := locals()) is not None:
                 #^^^^^^^^> {{locals() is assigned to "context" here.}}
       print("OK") 
    return render(request, "my_template.html", content_type=None, context=context) # Noncompliant {{Use an explicit context instead of passing "locals()" to this Django "render" call.}}
                                                                         #^^^^^^^

def using_locals_directly(request):
    username = "alice"
    password = "p@ssw0rd"
    return render(request, "my_template.html", locals()) # Noncompliant {{Use an explicit context instead of passing "locals()" to this Django "render" call.}}
                                              #^^^^^^^^

def using_locals_multiple_assignment(request):
    username = "alice"
    password = "p@ssw0rd"
    some_var = locals()
              #^^^^^^^^> {{locals() is assigned to "some_var" here.}}
    other_var = some_var
    my_context = other_var
    return render(request, "my_template.html", my_context) # Noncompliant {{Use an explicit context instead of passing "locals()" to this Django "render" call.}}
                                              #^^^^^^^^^^
def success(request):
    username = "alice"
    context = {username: username}
    return render(request, "my_template.html", context)

def success_not_passing_context(request):
    username = "alice"
    password = "p@ssw0rd"
    context = locals()
    return render(request, "my_template.html")

def success_render_dict_literal(request):
    username = "alice"
    password = "p@ssw0rd"
    context = locals()
    return render(request, "my_template.html", {})

def success_render_context_is_not_locals(request):
    username = "alice"
    password = "p@ssw0rd"
    context = foo()
    return render(request, "my_template.html", context)

def success_render_local_function(request):
    username = "alice"
    password = "p@ssw0rd"
    def cred():
      return {}
    return render(request, "my_template.html", cred())

def success_render_assigned_local_function(request):
    username = "alice"
    password = "p@ssw0rd"
    def cred():
      return {}
    context = cred()
    return render(request, "my_template.html", context)

def success_render_unknown_context_function(request):
    username = "alice"
    password = "p@ssw0rd"
    return render(request, "my_template.html", cred())

def success_render_unknown_context_param(request, ctx):
    username = "alice"
    password = "p@ssw0rd"
    return render(request, "my_template.html", ctx)

def success_render_global_context(request):
    username = "alice"
    password = "p@ssw0rd"
    return render(request, "my_template.html", context)

def success_render_not_called(request):
    username = "alice"
    password = "p@ssw0rd"
    context = locals()
    return foo(request, "my_template.html", context)

def success_multiple_assignment(request):
    username = "alice"
    password = "p@ssw0rd"
    some_var = {}
    other_var = some_var
    my_context = other_var
    return render(request, "my_template.html", my_context)

def success_too_many_assignment(request):
    username = "alice"
    password = "p@ssw0rd"
    a = locals()
    b = a
    c = b
    d = c
    e = d
    f = e
    g = f
    return render(request, "my_template.html", g)

from typing import Dict

def success_assignment_expression(request):
    if(context:= { "a": 1, "b" :2 }) is not None:
        print("ok")
    return render(request, "my_template.html", context=context)

def success_annotated_assignment(request):
    context: Dict[str, Any] = { "a": 1, "b" :2 }
    return render(request, "my_template.html", context=context)

def success_annotated_assignment_no_value(request):
    context: Dict[str, Any]
    return render(request, "my_template.html", context=context)
