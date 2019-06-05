from django.db import models
from django.db import connection
from django.db import connections
from django.db.models.expressions import RawSQL

class MyUser(models.Model):

    def query_my_user(request, params):
        hardcoded_request = 'SELECT * FROM mytable WHERE name = "test"'
        MyUser.objects.raw('SELECT * FROM mytable WHERE name = "%s"' % value)  # Noncompliant {{Make sure that executing SQL queries is safe here.}}
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        MyUser.objects.raw('SELECT * FROM mytable WHERE name = "%s"'.format(value))  # Noncompliant
        MyUser.objects.raw(f"SELECT * FROM mytable WHERE name = '{value}'")  # Noncompliant
        MyUser.objects.raw(F"SELECT * FROM mytable WHERE name = '{value}'")  # Noncompliant
        MyUser.objects.raw(request)  # OK
        MyUser.objects.raw(hardcoded_request)  # OK

        # Parametrized queries
        MyUser.objects.raw('SELECT * FROM mytable WHERE name = "%s"' + value, params) # Noncompliant

        with connection.cursor() as cursor:
            cursor.execute('SELECT * FROM mytable WHERE name = "%s"' % value)  # Noncompliant

        with connections['my_db'].cursor() as cursor:
            cursor.execute('SELECT * FROM mytable WHERE name = "%s"' % value)  # Noncompliant

        RawSQL("select col from mytable where mycol = %s", ("test",)) # OK

        MyUser.objects.extra(  # Noncompliant
            select={
                'mycol': "select col from sometable here mycol = %s and othercol = " + value
            },
        )
