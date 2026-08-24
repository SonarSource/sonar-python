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
        fieldNameWithPasswordInIt = os.getenv("password", "Azerty123")  # Noncompliant
        fieldNameWithPasswordInIt = os.environ.get("password", "Azerty123")  # Noncompliant
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
        password = "Xk28vQ91"                                     # Noncompliant
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
        pymysql.connect('localhost', 'root', 'Azerty123') # Noncompliant {{Review this potentially hard-coded credential.}}
#                                            ^^^^^^^^^^^
        pymysql.connections.Connection(host='localhost', user='root', password='password') # OK (avoid FPs)
        pymysql.connections.Connection('localhost', 'root', 'Azerty123') # Noncompliant
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
        pgdb.connect('localhost', 'postgres', 'Azerty123') # Noncompliant
        pgdb.connect(host='localhost', user='postgres', password=pwd) # OK
        pgdb.connect('localhost', 'postgres', pwd) # OK

        pg.DB(host='localhost', user='postgres', passwd='Azerty123') # Noncompliant
        pg.DB(None, 'localhost', 5432, None, 'postgres', 'Azerty123') # Noncompliant
        pg.DB(host='localhost', user='postgres', passwd=pwd) # OK
        pg.DB(None, 'localhost', 5432, None, 'postgres', pwd) # OK

        pg.connect(host='localhost', user='postgres', passwd='Azerty123') # Noncompliant
        pg.connect(None, 'localhost', 5432, None, 'postgres', 'Azerty123') # Noncompliant
        pg.connect(host='localhost', user='postgres', passwd=pwd) # OK
        pg.connect(None, 'localhost', 5432, None, 'postgres', pwd) # OK
        pg.connect(host='localhost', user='postgres', passwd='') # Compliant
        pg.connect(None, 'localhost', 5432, None, 'postgres', '') # Compliant

        random.call(None, password = 42) # OK
        random.call(None, password = "Azerty123") # Noncompliant
        random.call(None, password = "") # OK
        pg.connect(*unpack, 'localhost', 5432, None, 'postgres', pwd) # OK

    

class PASSWORD(A):
    def getPassword(self, password):
        pass
    def somePassword(self, password=42):  # OK
        pass
    def somePassword(self, password=""):  # OK
        pass
    def somePassword(self, *, password="Azerty123"): # Noncompliant
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
        'PASSWORD': os.getenv("DB_PASSWORD", "Azerty123"), # Noncompliant
        'PASSWORD': os.environ.get("DB_PASSWORD", "Azerty123"),  # Noncompliant
        'HOST': 'localhost',
        'PORT': '5432'
    },
    'any_other_key': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'quickdb',
        'USER': 'sonarsource',
        'PASSWORD': 'azerty123',                    # Noncompliant
        'PASSWORD': os.getenv('DB_PASSWORD'),       # Compliant
        'PASSWORD': os.getenv("DB_PASSWORD", "Azerty123"), # Noncompliant
        'PASSWORD': os.environ.get("DB_PASSWORD", "Azerty123"),  # Noncompliant
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
    app.config["SECRET_KEY"] = "my-lookup-key"  # Compliant, secret lookup key
    # SECRET_KEY holds the secret itself, so placeholder values are still a misconfiguration here,
    # unlike `password = "changeme"` in test_secret_classifier below.
    app.config["SECRET_KEY"] = "changeme"  # Noncompliant
    app.config["SECRET_KEY"] = "APP_SECRET_KEY"  # Compliant, environment-variable name
    # Environment-variable shape without a lookup token: not a lookup name.
    app.config["SECRET_KEY"] = "MY_APP_VALUE"  # Noncompliant
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")  # Compliant
    app.config["SECRET_KEY"] = os.getenv("APP_SECRET_KEY", "dev-only-placeholder")  # Compliant
    app.config["SECRET_KEY"] = os.urandom(24).hex()  # Compliant
    generated_secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(24).hex()
    app.config["SECRET_KEY"] = generated_secret_key  # Compliant
    hardcoded_secret_key = "hardcoded-value"
    app.config["SECRET_KEY"] = hardcoded_secret_key  # Noncompliant
    app.config["SECURITY_PASSWORD_HASH"] = "sha512_crypt"  # Compliant
    a, app.config["SECRET_KEY"] = "foo", "foo"  # Noncompliant
    app.config["SECURITY_PASSWORD_HASH"], app.config["SECRET_KEY"] = "foo", "foo"  # Noncompliant
    app.config["SECRET_KEY"], other = os.getenv("SECRET_KEY"), "foo"  # Compliant
    other, app.config["SECRET_KEY"] = "foo", os.getenv("SECRET_KEY")  # Compliant
    app.config["SECRET_KEY"], other = "foo", os.getenv("SECRET_KEY")  # Noncompliant
    other, app.config["SECRET_KEY"] = os.getenv("SECRET_KEY"), "foo"  # Noncompliant
    app.config["SECRET_KEY"] = (os.getenv("SECRET_KEY"), "foo")  # Noncompliant
    # Resolving a self-referential value must terminate instead of overflowing the stack.
    app.config["SECRET_KEY"] = recursive_value  # Noncompliant
    recursive_value = (recursive_value, "foo")
    if not app.config["SECRET_KEY"]:  # Compliant, reading the configured secret
        pass
    configured_secret_key = app.config["SECRET_KEY"]  # Compliant, reading the configured secret

def test_credential_identifiers_and_configuration_modes():
    password_policy = "strict"  # Compliant, configuration mode
    password_management = "external"  # Compliant, configuration mode
    settings = {"password_policy": "strict", "password_management": "default"}  # Compliant, configuration modes
    PASSWORD_PARAMETER_KEY = "alias/aws/ssm"  # Compliant, KMS key alias
    password_secret_name = os.getenv("PAYMENTS_SECRET_NAME", "payments-api-key")  # Compliant, secret lookup key
    password = "PAYMENTS_SECRET_NAME"  # Compliant, environment-variable name
    password = "payments-api-key"  # Compliant, secret lookup key
    password = "payments-secret-id"  # Compliant, secret lookup ID
    password = "payments-secret-name"  # Compliant, secret lookup name
    password_id = "primary"  # Compliant, credential metadata
    password_name = "primary"  # Compliant, credential metadata
    password = "azerty123"  # Noncompliant
    password_policy = "azerty123"  # Noncompliant
    password_management = "Azerty123"  # Noncompliant
    password_reference = "api-key-Azerty123"  # Noncompliant
    settings = {"password_policy": "azerty123"}  # Noncompliant
    # Generic, unreported metadata and identifier terms must not suppress issues.
    password_reference = "external"  # Noncompliant
    password_identifier = "strict"  # Noncompliant
    password = "external-ref"  # Noncompliant
    # An environment-variable shape alone is not enough: the value must name a credential slot.
    password = "PAYMENTS_SECRET_ID"  # Compliant, environment-variable name
    password = "HUNTER_2"  # Noncompliant
    password = "AZERTY_123"  # Noncompliant
    password_policy = "AZERTY_123"  # Noncompliant

def test_credential_identifiers_in_arguments(password="payments-api-key"):  # Compliant, secret lookup key
    connect(password="payments-api-key")  # Compliant, secret lookup key
    connect(password="azerty123")  # Noncompliant

def test_credential_identifiers_in_arguments_noncompliant(password="azerty123"):  # Noncompliant
    pass

def test_secret_classifier():
    # Values recognized by SecretClassifier as known non-secrets don't raise, even though the name matches.
    password = "changeme"                                   # Compliant, well-known placeholder secret
    password = "Xk28"                                        # Compliant, too short to be a real secret
    password = "${SECRET_KEY}"                               # Compliant, variable interpolation
    password = "op://vault/secret"                           # Compliant, external secret store reference
    password = "{cipher}1e3faa2cdab2deae117dca102e52922a"    # Compliant, encrypted marker
