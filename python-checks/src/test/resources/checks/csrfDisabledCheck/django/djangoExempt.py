def exemptDecoratorExample():
    from django.views.decorators.csrf import csrf_exempt
     
    @csrf_exempt # Noncompliant {{Make sure disabling CSRF protection is safe here.}}
    #^^^^^^^^^^^
    def csrftest2(request):
        pass
     
