"""
some docstring
password=hello
no issue
"""
from Crypto.Cipher import AES
import base64
import os

import mysql.connector
import pymysql
import psycopg2
import pgdb
import pg

from flask import Flask

# default words list: password,passwd,pwd,passphrase

secret_key = '1234567890123456'
something = something()
def getDecrypted(encodedtext):
    cipher = AES.new(secret_key, AES.MODE_ECB)
    return cipher.decrypt(base64.b64decode(encodedtext))

class A:
    """
    password=azerty123
    OK
    """
    passed = "passed"
    password = "azerty123" # Noncompliant
    password = "azerty123" # Noncompliant
    fieldNameWithPasswordInIt = "password" # OK
    fieldNameWithPasswordInIt = "" # OK
    user, password = get_credentials()
    (a, b) = ("some", "thing")

    def __init__(self):
        """
        password=azerty123
        OK
        """
        self.passed = "passed"
        fieldNameWithPasswordInIt = "azerty123"            # Noncompliant {{"password" detected here, review this potentially hard-coded credential.}}
        fieldNameWithPasswordInIt = os.getenv("password", "")  # OK
        self.fieldNameWithPasswordInIt = "azerty123"            # Noncompliant {{"password" detected here, review this potentially hard-coded credential.}}
        self.fieldNameWithPasswordInIt = os.getenv("password", "")  # OK

    def a(self,pwd="azerty123", other=None):  # Noncompliant {{"pwd" detected here, review this potentially hard-coded credential.}}

        var1 = 'admin'
        var1 = 'user=admin&password=Azerty123'        # Noncompliant {{"password" detected here, review this potentially hard-coded credential.}}
        var1 = 'user=admin&passwd=Azerty123'          # Noncompliant {{"passwd" detected here, review this potentially hard-coded credential.}}
        var1 = 'user=admin&pwd=Azerty123'             # Noncompliant {{"pwd" detected here, review this potentially hard-coded credential.}}
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
        var1 = f"&password={pwd}"                       # OK
        var1 = "&password={something}"                  # OK

        url = "http://user:azerty123@domain.com"      # Noncompliant {{Review this hard-coded URL, which may contain a credential.}}
        url = "https://user:azerty123@domain.com"      # Noncompliant {{Review this hard-coded URL, which may contain a credential.}}
        url = "ftp://user:azerty123@domain.com"      # Noncompliant {{Review this hard-coded URL, which may contain a credential.}}
        url = "http://user:@domain.com"               # OK
        url = "http://user@domain.com:80"             # OK
        url = "http://user@domain.com"                # OK
        url = "http://domain.com/user:azerty123"      # OK
        url = "ssh://domain.com/user:azerty123"      # OK
        url = "unknown://domain.com/user:azerty123"      # OK

        username = 'admin'        
        password = pwd
        password = 'azerty123'                                    # Noncompliant {{"password" detected here, review this potentially hard-coded credential.}}
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

        # To avoid FPs, no issues raised on equality tests
        if password == 'Azerty123': # OK
            pass
        elif password.__eq__('Azerty123'): # OK
            pass
        elif 'Azerty123'.__eq__(password): # OK
            pass

        hash_map = { 'password': "azerty123"} # Noncompliant {{"password" detected here, review this potentially hard-coded credential.}}
        hash_map = { ("a", "b") : "c"} # OK
        hash_map = { something : "c"} # OK
        hash_map = {'admin_form' : adminForm, **self.admin.context(request),} # OK
        hash_map = { 'password': pwd} # OK
        hash_map = { 'password': "password"} # OK
        hash_map['db_password'] = "azerty123" # Noncompliant
        hash_map['db_password'] = pwd # OK
        hash_map['something'] = "azerty123" # OK
        hash_map[something] = "something" # OK
        hash_map['password'] = 'password' # OK

        encoded_user = 'gUhd9TxpnQppnZVAf7cv9pa5sgRo2sFmShrr/NK9dz0='
        encoded_password = 'gUhd9TxpnQppnZVAf7cv9uVnoE28Vq0bR2Cx6Ku1UQA=' # Noncompliant
        username = getDecrypted(encoded_user)                       
        password = getDecrypted(encoded_password)                   # OK
    
    def db(self, pwd):
        mysql.connector.connect(host='localhost', user='root', password='Azerty123')  # Noncompliant
        mysql.connector.connection.MySQLConnection(host='localhost', user='root', password='password')  # OK (avoid FPs)
        mysql.connector.connect(host='localhost', user='root', password=pwd)  # OK
        mysql.connector.connection.MySQLConnection(host='localhost', user='root', password=pwd)  # OK
        mysql.connector.connection.MySQLConnection(host='localhost', user='root', password='')  # OK
        mysql.connector.connection.MySQLConnection(host='localhost', user='root', "")  # OK

        pymysql.connect(host='localhost', user='root', password='Azerty123') # Noncompliant
        pymysql.connect('localhost', 'root', 'password') # Noncompliant {{Review this potentially hard-coded credential.}}
#                                            ^^^^^^^^^^
        pymysql.connections.Connection(host='localhost', user='root', password='password') # OK (avoid FPs)
        pymysql.connections.Connection('localhost', 'root', 'password') # Noncompliant
        pymysql.connect(host='localhost', user='root', password=pwd) # OK
        pymysql.connect('localhost', 'root', pwd) # OK
        pymysql.connections.Connection(host='localhost', user='root', password=pwd) # OK
        pymysql.connections.Connection('localhost', 'root', pwd) # OK
        pymysql.connect('localhost', 'root', '') # Compliant
        pymysql.connect(host='localhost', user='root', password='') # Compliant
        pymysql.connections.Connection(host='localhost', user='root', password='') # Compliant
        pymysql.connections.Connection('localhost', 'root', '') # Compliant

        psycopg2.connect(host='localhost', user='postgres', password='Azerty123') # Noncompliant
        psycopg2.connect(host='localhost', user='postgres', password=pwd,) # OK

        pgdb.connect(host='localhost', user='postgres', password='Azerty123') # Noncompliant
        pgdb.connect('localhost', 'postgres', 'password') # Noncompliant
        pgdb.connect(host='localhost', user='postgres', password=pwd) # OK
        pgdb.connect('localhost', 'postgres', pwd) # OK

        pg.DB(host='localhost', user='postgres', passwd='Azerty123') # Noncompliant
        pg.DB(None, 'localhost', 5432, None, 'postgres', 'password') # Noncompliant
        pg.DB(host='localhost', user='postgres', passwd=pwd) # OK
        pg.DB(None, 'localhost', 5432, None, 'postgres', pwd) # OK

        pg.connect(host='localhost', user='postgres', passwd='Azerty123') # Noncompliant
        pg.connect(None, 'localhost', 5432, None, 'postgres', 'password') # Noncompliant
        pg.connect(host='localhost', user='postgres', passwd=pwd) # OK
        pg.connect(None, 'localhost', 5432, None, 'postgres', pwd) # OK
        pg.connect(host='localhost', user='postgres', passwd='') # Compliant
        pg.connect(None, 'localhost', 5432, None, 'postgres', '') # Compliant

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

#To avoid false positives, no issue is raised when a credential word is present both as a key/variable name and as a value
dict1 = {'password': ''} # Compliant
dict2 = dict(password='AZURE_PASSWORD') # Compliant
dict3 = {'password': 'password'} # Compliant
dict4 = {"login_password": "password"} # Compliant
module.fail_json(msg="Password parameter is missing."
                                     " Please specify this parameter in task or"
                                     " export environment variable like 'export VMWARE_PASSWORD=ESXI_PASSWORD'") # Compliant
jim = User(username='jimcarry',password="password88") # Compliant
conn = pymssql.connect(server='yourserver', user='yourusername@yourserver',
             password='yourpassword', database='yourdatabase') # Compliant

def test_flask():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "foo"  # Noncompliant
    app.config["SECURITY_PASSWORD_HASH"] = "sha512_crypt"  # Compliant
    a, app.config["SECRET_KEY"] = "foo", "foo"  # Noncompliant
    app.config["SECURITY_PASSWORD_HASH"], app.config["SECRET_KEY"] = "foo", "foo"  # Noncompliant
