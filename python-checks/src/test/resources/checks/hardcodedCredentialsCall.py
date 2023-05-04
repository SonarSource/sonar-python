import pylast
import psycopg2cffi
import pymysql.tests.test_connection as test_connection

def obj_creation_hardcoded_credentials():
    network = pylast.LibreFMNetwork("api_key", # Noncompliant {{Revoke and change this password, as it is compromised.}}
#                                   ^^^^^^^^^ 0
                                    "api_secret", # Noncompliant
                                    "session_key",
                                    "username",
                                    "password_hash") # Noncompliant

def obj_creation_empty_credentials():
    network = pylast.LibreFMNetwork("", "", "", "", "")
    temp_user = test_connection.TempUser(password="")
    password = ""
    temp_user = test_connection.TempUser(password=password)

def obj_creation_hardcoded_credentials_named_arguments():
    network = pylast.LibreFMNetwork(api_key="api_key", # Noncompliant {{Revoke and change this password, as it is compromised.}}
#                                   ^^^^^^^^^^^^^^^^^ 0
                                    api_secret="api_secret", # Noncompliant
                                    session_key="session_key",
                                    username="username",
                                    password_hash="password_hash") # Noncompliant

    temp_user = test_connection.TempUser(password="password") # Noncompliant


def obj_creation_hardcoded_assigned_value(password):
    temp_user = test_connection.TempUser(password=password)
    temp_user = test_connection.TempUser(password=foo())

    api_key = "abc"
#             ^^^^^> 1 {{Revoke and change this password, as it is compromised.}}
    network = pylast.LibreFMNetwork(api_key=api_key) # Noncompliant
#                                   ^^^^^^^^^^^^^^^ 1

    value = "abcd"
#           ^^^^^^> 1 {{Revoke and change this password, as it is compromised.}}
    api_secret = value
    network = pylast.LibreFMNetwork(api_secret=api_secret) # Noncompliant
#                                   ^^^^^^^^^^^^^^^^^^^^^ 1


def top_level_function_call():
    psycopg2cffi.connect(password="abc") # Noncompliant

# from S4433
import ldap
import os

def ldap_methods_check(p):
    connect = ldap.initialize('ldap://127.0.0.1:1389')
    connect.set_option(ldap.OPT_REFERRALS, 0)

    connect.simple_bind('cn=root')
    connect.simple_bind_s('cn=root')
    connect.bind_s('cn=root', None)
    connect.bind('cn=root', None)
    connect.bind('cn=root', "")
    connect.bind('cn=root', cred="")
    connect.bind('cn=root', serverctrls=os.environ.get('LDAP_PASSWORD'))

    pwd = ""
    connect.bind('cn=root', pwd)
    connect.simple_bind()
    args = ['cn=root']
    connect.simple_bind(*args)

# from S2115
import mysql.connector
import pymysql
import psycopg2
import pgdb
import pg

def db_connect(pwd):
    mysql.connector.connect(host='localhost', user='sonarsource', password='')
    mysql.connector.connect(host='localhost', password='', user='sonarsource')
    mysql.connector.connect('localhost', 'sonarsource', '')

    mysql.connector.connection.MySQLConnection(host='localhost', user='sonarsource', password='')
    pymysql.connect(host='localhost', user='sonarsource', password='')
    pymysql.connections.Connection(host='localhost', user='sonarsource', password='abc') # Noncompliant
    psycopg2.connect(host='localhost', user='postgres', password='')
    pgdb.connect(host='localhost', user='postgres', password='')

    pg.DB(host='localhost', user='postgres', passwd='')
    pg.DB('dbname', 'localhost', 5432, 'opt', 'postgres', '')
    pg.connect(host='localhost', user='postgres', passwd='')


def byte_decode_hardcoded_value():
    apiKey = b"abc".decode("utf-8")
#            ^^^^^^> 1 {{Revoke and change this password, as it is compromised.}}
    network = pylast.LibreFMNetwork(api_key = apiKey) # Noncompliant {{Revoke and change this password, as it is compromised.}}
#                                   ^^^^^^^^^^^^^^^^ 1

def byte_decode_hardcoded_value():
    b = apiKey
    apiKey = b
    network = pylast.LibreFMNetwork(api_key = apiKey)
