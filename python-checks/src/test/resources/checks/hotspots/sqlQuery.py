from django.db import models
from django.db import connection
from django.db import connections
from django.db.models.expressions import RawSQL

class MyUser(models.Model):

    def query_my_user(request, params):
        hardcoded_request = 'SELECT * FROM mytable WHERE name = "test"'
        formatted_request = 'SELECT * FROM mytable WHERE name = "%s"' % value
        formatted_request2 = f'SELECT * FROM mytable WHERE name = "{value}"'
        formatted_request3 = F'SELECT * FROM mytable WHERE name = "{value}"'
        formatted_request4 = 'SELECT * FROM mytable WHERE name = "%s"' % value
        if request:
            formatted_request4 = 'SELECT * FROM mytable WHERE name = "%s"' % value
        global formatted_request5
        MyUser.objects.raw('SELECT * FROM mytable WHERE name = "%s"' % value)  # Noncompliant {{Make sure that formatting this SQL query is safe here.}}
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        MyUser.objects.raw('SELECT * FROM mytable WHERE name = "%s"'.format(value))  # Noncompliant
        MyUser.objects.raw(f"SELECT * FROM mytable WHERE name = '{value}'")  # Noncompliant
        MyUser.objects.raw(F"SELECT * FROM mytable WHERE name = '{value}'")  # Noncompliant
        MyUser.objects.raw(x := F"SELECT * FROM mytable WHERE name = '{value}'")  # Noncompliant
        MyUser.objects.raw(request)  # OK
        MyUser.objects.raw(hardcoded_request)  # OK
        MyUser.objects.raw(x := hardcoded_request)  # OK
        MyUser.objects.raw(formatted_request)  #  Noncompliant [[secondary=-14]]
        MyUser.objects.raw(formatted_request2)  # Noncompliant
        MyUser.objects.raw(formatted_request3)  # Noncompliant
        MyUser.objects.raw(y := formatted_request3)  # Noncompliant
        MyUser.objects.raw((y := formatted_request3))  # Noncompliant
        MyUser.objects.raw(formatted_request4)  # FN, multiple assignments
        MyUser.objects.raw(formatted_request5)  # OK
        MyUser.objects.raw(*formatted_request5)  # OK

        # Parametrized queries
        MyUser.objects.raw('SELECT * FROM mytable WHERE name = "%s"' + value, params) # Noncompliant

        with connection.cursor() as cursor:
            cursor.execute('SELECT * FROM mytable WHERE name = "%s"' % value)  # Noncompliant

        with connections['my_db'].cursor() as cursor:
            cursor.execute('SELECT * FROM mytable WHERE name = "%s"' % value)  # Noncompliant

        RawSQL("select col from mytable where mycol = %s", ("test",)) # OK
        RawSQL('SELECT * FROM mytable WHERE name = "%s"' % value) # Noncompliant
        sql_query = 'SELECT * FROM mytable WHERE name = "%s"' % value
        RawSQL(sql_query) # Noncompliant
        RawSQL() # OK

        MyUser.objects.extra(  # Noncompliant
            select={
                'mycol': "select col from sometable here mycol = %s and othercol = " + fun()
            },
        )

        MyUser.objects.extra({ 'mycol': "select col from sometable here mycol = %s and othercol = " + value}) # Noncompliant
        MyUser.objects.extra({ 'mycol': "select col from sometable here mycol = %s and othercol = " + ""}) # Noncompliant

     def fun():
        pass
