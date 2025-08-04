import boto3

import pymysql.connect
import mysql.connector
import psycopg2
import sqlite3
import redis
import peewee
import mongoengine
import pymongo

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def lambda_handler(event, context):
    s3_client = boto3.client('s3')  # Noncompliant {{Initialize this AWS client outside the Lambda handler function.}}
#               ^^^^^^^^^^^^^^^^^^

    s3 = boto3.resource('s3') # FN SONARPY-3224
    session = boto3.session.Session()  # FN SONARPY-3224

    pymysql_connection = pymysql.connect(host='localhost')  # Noncompliant {{Initialize this database connection outside the Lambda handler function.}}
    #                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    mysql_connection = mysql.connector.connect(host="localhost")  # Noncompliant
    mongo_client = pymongo.MongoClient()  # Noncompliant
    sqlite3_connection = sqlite3.connect("tutorial.db") # Noncompliant
    # psycopg2_connection = psycopg2.connect("localhost") # FN SONARPY-3224
    redis_client = redis.Redis(host='localhost', port=6379, db=0) # FN SONARPY-3224
    strict_redis_client = redis.StrictRedis(host='localhost', port=6379, db=0) # FN SONARPY-3224

    engine = create_engine("postgresql+psycopg2://scott:tiger@localhost/")
    sqlalchemy_session = sessionmaker(engine)  # Noncompliant {{Initialize this ORM connection outside the Lambda handler function.}}
    #                    ^^^^^^^^^^^^^^^^^^^^

    peewee_sqlite_db = peewee.SqliteDatabase('/path/to/app.db', pragmas={'journal_mode': 'wal', 'cache_size': -1024 * 64})  # Noncompliant
    peewee_mysql_db = peewee.MySQLDatabase('my_app', user='app', password='db_password', host='10.1.0.8', port=3306)  # Noncompliant
    peewee_pg_db = peewee.PostgresqlDatabase('my_app', user='postgres', password='secret', host='10.1.0.9', port=5432)  # Noncompliant

    mongoengine_connection = mongoengine.connect('project1')  # Noncompliant


s3_client = boto3.client('s3')
s3 = boto3.resource('s3')
session = boto3.session.Session()
pymysql_connection = pymysql.connect(host='localhost')
mysql_connection = mysql.connector.connect(host="localhost")
mongo_client = pymongo.MongoClient()

psycopg2_connection = psycopg2.connect("localhost")
sqlite3_connection = sqlite3.connect("tutorial.db")
redis_client = redis.Redis(host='localhost', port=6379, db=0)
strict_redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

engine = create_engine("postgresql+psycopg2://scott:tiger@localhost/")
sqlalchemy_session = sessionmaker(engine)

peewee_sqlite_db = peewee.SqliteDatabase('/path/to/app.db', pragmas={'journal_mode': 'wal', 'cache_size': -1024 * 64})
peewee_mysql_db = peewee.MySQLDatabase('my_app', user='app', password='db_password', host='10.1.0.8', port=3306)
peewee_pg_db = peewee.PostgresqlDatabase('my_app', user='postgres', password='secret', host='10.1.0.9', port=5432)

mongoengine_connection = mongoengine.connect('project1')
