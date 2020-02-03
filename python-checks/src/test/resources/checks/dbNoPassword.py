import mysql.connector
import pymysql
import psycopg2
import pgdb
import pg

DATABASES = {
    'postgresql_db': {
        'PASSWORD': '',   # Ok, not in Django "settings.py" file
    },
}

# SQLAlchemy & Flask-SQLAlchemy
def configure_app(flask_app, pwd):
    db_uri = "postgresql://user:@domain.com" # Noncompliant {{Add password protection to this database.}}
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    db_uri = "postgresql://user@domain.com" # Noncompliant
    db_uri = "foopostgresql://user:@domain.com" # ok
    db_uri = "postgre://user:@domain.com"       # ok

    db_uri = "postgresql://domain.com"
    db_uri = "postgresql://domain.com:"
    db_uri = "postgresql://domain.com:5432"
    db_uri = "postgresql://domain.com?user=user:"
    db_uri = f"postgresql://user:{pwd}@domain.com"
    db_uri = "postgresql://user:%s@domain.com".format(pwd)
    db_uri = "postgresql://user:%s@domain.com" % pwd

    # URL schema to match. The engine specifier between "+" and "://" can be any alphanumeric sequence
    db_uri = "postgresql+engine://user:@domain.com" # Noncompliant
    db_uri = "postgresql+engine://user:psw@domain.com" # OK
    db_uri = "mysql://user:@domain.com" # Noncompliant
    db_uri = "mysql+engine://user:@domain.com"  # Noncompliant
    db_uri = "oracle://user:@domain.com" # Noncompliant
    db_uri = "oracle+engine://user:@domain.com" # Noncompliant
    db_uri = "mssql://user:@domain.com" # Noncompliant
    db_uri = "mssql+engine42://user:@domain.com" # Noncompliant


def db_connect(pwd):
    mysql.connector.connect(host='localhost', user='sonarsource', password='')  # Noncompliant
#                                                                 ^^^^^^^^^^^
    mysql.connector.connect(host='localhost', password='', user='sonarsource')  # Noncompliant
    mysql.connector.connect('localhost', 'sonarsource', '')  # Noncompliant
#                                                       ^^
    mysql.connector.connect('localhost', 'sonarsource', 'hello')  # OK
    mysql.connector.connect('localhost', 'sonarsource', password='hello')  # OK
    mysql.connector.connect('localhost', 'sonarsource', password=pwd)  # OK
    mysql.connector.connect('localhost', 'sonarsource', pwd)  # OK
    mysql.connector.connect('localhost', 'sonarsource')           # OK
    mysql.connector.connect('localhost', 'sonarsource', **dict)   # OK

    mysql.connector.connection.MySQLConnection(host='localhost', user='sonarsource', password='')  # Noncompliant
    pymysql.connect(host='localhost', user='sonarsource', password='') # Noncompliant
    pymysql.connections.Connection(host='localhost', user='sonarsource', password='') # Noncompliant
    psycopg2.connect(host='localhost', user='postgres', password='') # Noncompliant
    pgdb.connect(host='localhost', user='postgres', password='') # Noncompliant

    pg.DB(host='localhost', user='postgres', passwd='')       # Noncompliant
    pg.DB('dbname', 'localhost', 5432, 'opt', 'postgres', '') # Noncompliant
    pg.connect(host='localhost', user='postgres', passwd='')  # Noncompliant
    pg.DB(host='localhost', user='postgres', passwd=pwd)       # Compliant
    pg.DB('dbname', 'localhost', 5432, 'opt', 'postgres', pwd) # Compliant
