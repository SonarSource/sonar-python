from django.shortcuts import render

def using_locals_from_var(request):
    username = "alice"
    password = "p@ssw0rd"
    context = locals()
             #^^^^^^^^> {{locals() is assigned to "context" here.}}
    return render(request, "my_template.html", context) # Noncompliant {{Use an explicit context instead of passing "locals()" to this Django "render" call.}} [[secondary=-1]]
                                              #^^^^^^^
def using_locals_with_named_params(request):
    username = "alice"
    password = "p@ssw0rd"
    context = locals()
             #^^^^^^^^> {{locals() is assigned to "context" here.}}
    return render(request, "my_template.html", content_type=None, context=context) # Noncompliant {{Use an explicit context instead of passing "locals()" to this Django "render" call.}} [[secondary=-1]]
                                                                         #^^^^^^^
def using_locals_directly(request):
    username = "alice"
    password = "p@ssw0rd"
    return render(request, "my_template.html", locals()) # Noncompliant {{Use an explicit context instead of passing "locals()" to this Django "render" call.}}
                                              #^^^^^^^^

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
