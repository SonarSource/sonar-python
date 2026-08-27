import not_oracledb

cursor.execute(f"SELECT * FROM customers WHERE id = '{value}'")  # OK

from not_oracledb import connect

cursor.execute(f"SELECT * FROM customers WHERE id = '{value}'")  # OK

# Importing OracleDB connection factories without calling them must not taint the file.
from oracledb import connect
from cx_Oracle import connect as cx_connect

cursor.execute(f"SELECT * FROM customers WHERE id = '{value}'")  # OK

# Importing the oracledb package without using oracledb.connect (e.g. only for exception
# handling) must not be treated as a signal that this file uses OracleDB.
import oracledb

try:
    pass
except oracledb.DatabaseError:
    pass

cursor.execute(f"SELECT * FROM customers WHERE id = '{value}'")  # OK

# Same as above, for the cx_Oracle legacy package.
import cx_Oracle

try:
    pass
except cx_Oracle.DatabaseError:
    pass

cursor.execute(f"SELECT * FROM customers WHERE id = '{value}'")  # OK

# Conflicting imports bound to the same name give that symbol a null FQN; the check
# must not crash on it.
def run_query_with_fallback_driver(value):
    try:
        import cx_Oracle as db
    except ImportError:
        import sqlite3 as db

    db.connect()
    cursor.execute(f"SELECT * FROM customers WHERE id = '{value}'")  # OK
