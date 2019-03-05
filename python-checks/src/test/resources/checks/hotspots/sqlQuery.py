from django.db import models
from django.db import connection
from django.db import connections
from django.db.models.expressions import RawSQL

class MyUser(models.Model):

    def query_my_user(request, params):
        hardcoded_request = 'SELECT * FROM mytable WHERE name = "test"'
        formatted_request = 'SELECT * FROM mytable WHERE name = "%s"' % value
        MyUser.objects.raw(request)  # Noncompliant {{Make sure that executing SQL queries is safe here.}}
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        MyUser.objects.raw(formatted_request)  # Noncompliant
        MyUser.objects.raw(hardcoded_request)  # Noncompliant, FP: limitation on hardcoded string

        # Parametrized queries
        MyUser.objects.raw(request, params) # Noncompliant

        with connection.cursor() as cursor:
            cursor.execute(request)  # Noncompliant

        with connections['my_db'].cursor() as cursor:
            cursor.execute(request)  # Noncompliant

        RawSQL("select col from mytable where mycol = %s", ("test",)) # Noncompliant

        MyUser.objects.extra(  # Noncompliant
            select={
                'mycol': 'myothercol > 10'
            },
        )
