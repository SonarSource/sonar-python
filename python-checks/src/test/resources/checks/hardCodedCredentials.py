from Crypto.Cipher import AES
import base64
import os

import mysql.connector
import pymysql
import psycopg2
import pgdb
import pg

# default words list: password,passwd,pwd,passphrase

secret_key = '1234567890123456'
something = something()
def getDecrypted(encodedtext):
    cipher = AES.new(secret_key, AES.MODE_ECB)
    return cipher.decrypt(base64.b64decode(encodedtext))

class A:

    passed = "passed"
    password = "azerty123" # Noncompliant
    password = "azerty123" # Noncompliant
    fieldNameWithPasswordInIt = "password" # OK
    fieldNameWithPasswordInIt = "" # OK
    user, password = get_credentials()
    (a, b) = ("some", "thing")

    def __init__(self):
        self.passed = "passed"
        fieldNameWithPasswordInIt = "azerty123"            # Noncompliant
        fieldNameWithPasswordInIt = os.getenv("password", "")  # OK
        self.fieldNameWithPasswordInIt = "azerty123"            # Noncompliant
        self.fieldNameWithPasswordInIt = os.getenv("password", "")  # OK

    def a(self,pwd="azerty123", other=None):  # Noncompliant

        var1 = 'admin'
        var1 = 'user=admin&password=Password123'        # Noncompliant
        var1 = 'user=admin&passwd=Password123'          # Noncompliant
        var1 = 'user=admin&pwd=Password123'             # Noncompliant
        var1 = 'user=admin&password='                   # OK
        var1 = 'user=admin&password= '                  # OK
        var1 = "user=%s&password=%s" % "Password123"    # OK FN?
        var1 = "user=%s&password=%s" % pwd              # OK
        var1 = f"&password={pwd}"                       # OK
        var1 = f"&password='{pwd}'"                     # OK
        var1 = "&password=?"                            # OK
        var1 = "&password=:password"                    # OK
        var1 = "&password=:param"                       # OK
        var1 = "&password='"+pwd+"'"                    # OK

        url = "http://user:azerty123@domain.com"      # Noncompliant
        url = "http://user:@domain.com"               # OK
        url = "http://user@domain.com:80"             # OK
        url = "http://user@domain.com"                # OK
        url = "http://domain.com/user:azerty123"      # OK

        username = 'admin'        
        password = pwd
        password = 'azerty123'                                    # Noncompliant
        password = "azerty123"                                    # Noncompliant
        password = '''azerty123'''                                # Noncompliant
        password = """azerty123"""                                # Noncompliant
        password = u'azerty123'                                   # Noncompliant
        password = f"azerty123"                                   # Noncompliant
        password = b"azerty123"                                   # Noncompliant
        password = "?"                                            # Noncompliant
        variableNameWithPasswordInIt = 'azerty123'                # Noncompliant
        variableNameWithPassphraseInIt = 'azerty123'              # Noncompliant
        variableNameWithPasswdInIt ='azerty123'                   # Noncompliant
        variableNameWithPwdInIt ='azerty123'                      # Noncompliant
        variableNameWithPasswordInItEmpty = ""                    # OK

        # To avoid FP due to RSPEC-1192 String literals should not be duplicated
        # Don't raise if the word is present both in the varariable and in the litteral string
        json_password = "password"                                # OK
        pwd = "pwd"                                               # OK
        PASSWORD = "Password"                                     # OK
        PASSWORD_INPUT = "[id='password']"                        # OK
        PASSWORD_PROPERTY = "custom.password"                     # OK
        TRUSTSTORE_PASSWORD = "trustStorePassword"                # OK
        CONNECTION_PASSWORD = "connection.password"               # OK
        RESETPWD = "/users/resetUserPassword"                     # OK

        if password == 'Password123': # Noncompliant
            pass
        elif 'Password123' == password: # Noncompliant
            pass
        elif password.__eq__('Password123'): # Noncompliant
            pass
        elif something.__eq__('Password123'): # OK
             pass
        elif password.__eq__(something): # OK
            pass
        elif password.__eq__(*unpack): # OK
            pass
        elif 'Password123'.__eq__(password): # Noncompliant
            pass
        elif 'Password123'.__eq__(something): # OK
            pass
        elif 'Password123'.__eq__(*unpack): # OK
            pass
        elif 'Password123'.__eq__("something"): # OK
            pass
        if password == None: # OK (FN?)
            pass
        if something == "something": # OK
            pass
        if password == '': # Noncompliant
            pass
        if password == "": # Noncompliant
            pass
        if password == pwd: # OK (FN?)
            pass
        if something.password == "Azerty123": # Noncompliant
            pass
        if foo.bar == "Azerty123": # OK
            pass

        hash_map = { 'password': "azerty123"} # Noncompliant
        hash_map = { ("a", "b") : "c"} # OK
        hash_map = { something : "c"} # OK
        hash_map = {'admin_form' : adminForm, **self.admin.context(request),} # OK
        hash_map['db_password'] = "azerty123" # Noncompliant
        hash_map['db_password'] = pwd # OK
        hash_map['something'] = "azerty123" # OK
        hash_map[something] = "something" # OK

        encoded_user = 'gUhd9TxpnQppnZVAf7cv9pa5sgRo2sFmShrr/NK9dz0='
        encoded_password = 'gUhd9TxpnQppnZVAf7cv9uVnoE28Vq0bR2Cx6Ku1UQA=' # Noncompliant
        username = getDecrypted(encoded_user)                       
        password = getDecrypted(encoded_password)                   # OK
    
    def db(self, pwd):
        mysql.connector.connect(host='localhost', user='root', password='password')  # Noncompliant
        mysql.connector.connection.MySQLConnection(host='localhost', user='root', password='password')  # Noncompliant
        mysql.connector.connect(host='localhost', user='root', password=pwd)  # OK
        mysql.connector.connection.MySQLConnection(host='localhost', user='root', password=pwd)  # OK

        pymysql.connect(host='localhost', user='root', password='password') # Noncompliant
        pymysql.connect('localhost', 'root', 'password') # Noncompliant
        pymysql.connections.Connection(host='localhost', user='root', password='password') # Noncompliant
        pymysql.connections.Connection('localhost', 'root', 'password') # Noncompliant
        pymysql.connect(host='localhost', user='root', password=pwd) # OK
        pymysql.connect('localhost', 'root', pwd) # OK
        pymysql.connections.Connection(host='localhost', user='root', password=pwd) # OK
        pymysql.connections.Connection('localhost', 'root', pwd) # OK

        psycopg2.connect(host='localhost', user='postgres', password='password') # Noncompliant
        psycopg2.connect(host='localhost', user='postgres', password=pwd,) # OK

        pgdb.connect(host='localhost', user='postgres', password='password') # Noncompliant
        pgdb.connect('localhost', 'postgres', 'password') # Noncompliant
        pgdb.connect(host='localhost', user='postgres', password=pwd) # OK
        pgdb.connect('localhost', 'postgres', pwd) # OK

        pg.DB(host='localhost', user='postgres', passwd='password') # Noncompliant
        pg.DB(None, 'localhost', 5432, None, 'postgres', 'password') # Noncompliant
        pg.DB(host='localhost', user='postgres', passwd=pwd) # OK
        pg.DB(None, 'localhost', 5432, None, 'postgres', pwd) # OK

        pg.connect(host='localhost', user='postgres', passwd='password') # Noncompliant
        pg.connect(None, 'localhost', 5432, None, 'postgres', 'password') # Noncompliant
        pg.connect(host='localhost', user='postgres', passwd=pwd) # OK
        pg.connect(None, 'localhost', 5432, None, 'postgres', pwd) # OK

        random.call(None, password = 42) # OK
        random.call(None, password = "hello") # Noncompliant
        random.call(None, password = "") # OK
        pg.connect(*unpack, 'localhost', 5432, None, 'postgres', pwd) # OK

    

class PASSWORD(A):
    def getPassword(self, password):
        pass
    def somePassword(self, password=42):  # OK
        pass
    def somePassword(self, password=""):  # OK
        pass
    def somePassword(self, *, password="hello"): # Noncompliant
        pass

instance = A()
instance.db('password')

DATABASES = {
    'postgresql_db': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'quickdb',
        'USER': 'sonarsource',
        'PASSWORD': 'azerty123',                    # Noncompliant
        'PASSWORD': os.getenv('DB_PASSWORD'),       # Compliant
        'HOST': 'localhost',
        'PORT': '5432'
    },
    'any_other_key': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'quickdb',
        'USER': 'sonarsource',
        'PASSWORD': 'azerty123',                    # Noncompliant
        'PASSWORD': os.getenv('DB_PASSWORD'),       # Compliant
        'HOST': 'localhost',
        'PORT': '5432'
    }
}
