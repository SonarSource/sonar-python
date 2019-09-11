class MyUser(models.Model):

    def query_my_user(request, params):
        hardcoded_request = 'SELECT * FROM mytable WHERE name = "test"'
        MyUser.objects.raw('SELECT * FROM mytable WHERE name = "%s"' % value)  #

        MyUser.objects.raw('SELECT * FROM mytable WHERE name = "%s"'.format(value))  #  
        MyUser.objects.raw(f"SELECT * FROM mytable WHERE name = '{value}'")  #  
        MyUser.objects.raw(F"SELECT * FROM mytable WHERE name = '{value}'")  #  
        MyUser.objects.raw(request)  # OK
        MyUser.objects.raw(hardcoded_request)  # OK

        # Parametrized queries
        MyUser.objects.raw('SELECT * FROM mytable WHERE name = "%s"' + value, params) #  

        with connection.cursor() as cursor:
            cursor.execute('SELECT * FROM mytable WHERE name = "%s"' % value)  #  

        with connections['my_db'].cursor() as cursor:
            cursor.execute('SELECT * FROM mytable WHERE name = "%s"' % value)  #  

        RawSQL("select col from mytable where mycol = %s", ("test",)) # OK

        MyUser.objects.extra(  #  
            select={
                'mycol': "select col from sometable here mycol = %s and othercol = " + fun()
            },
        )

        MyUser.objects.extra({ 'mycol': "select col from sometable here mycol = %s and othercol = " + value}) #  
        MyUser.objects.extra({ 'mycol': "select col from sometable here mycol = %s and othercol = " + ""}) #  

