def stdlib_tests(param):
    from http.cookies import SimpleCookie

    cookie = SimpleCookie()

    cookie['c1'] = 'value' # FN

    cookie['c2'] = 'value'
    cookie['c2']['secure'] = False # Noncompliant {{Make sure creating this cookie without the "secure" flag is safe.}}
    cookie['c2']['other'] = False # OK
    cookie['c2'][param] = False # OK

    cookie['c2']['secure', 'other'] = False # OK

    cookie['secure'] = False # OK

    cookie['c3'] = 'value'
    cookie['c3']['secure'] = True # OK

    not_a_cookie = 42
    not_a_cookie['c']['secure'] = False # OK

    get_cookie()['c']['secure'] = False # OK

def get_cookie(): pass
