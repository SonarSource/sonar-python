import oracledb
import oracledb as db
import cx_Oracle


def run_query(value):
    connection = oracledb.connect(user="scott", password="tiger", dsn="localhost/orclpdb")
    cursor = connection.cursor()
    cursor.execute(f"SELECT * FROM customers WHERE id = '{value}'")  # Noncompliant {{Make sure that formatting this SQL query is safe here.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    cursor.execute("SELECT * FROM customers WHERE id = :id")  # OK

    cursor.executemany(f"SELECT * FROM customers WHERE id = '{value}'", [])  # Noncompliant
    cursor.parse(f"SELECT * FROM customers WHERE id = '{value}'")  # Noncompliant
    cursor.prepare(f"SELECT * FROM customers WHERE id = '{value}'")  # Noncompliant
    cursor.fetch_df_all(f"SELECT * FROM customers WHERE id = '{value}'")  # Noncompliant
    cursor.fetch_df_batches(f"SELECT * FROM customers WHERE id = '{value}'")  # Noncompliant


def run_query_with_alias(value):
    connection = db.connect(user="scott", password="tiger", dsn="localhost/orclpdb")
    cursor = connection.cursor()
    cursor.execute(f"SELECT * FROM customers WHERE id = '{value}'")  # Noncompliant


def run_query_cx_oracle_module(value):
    connection = cx_Oracle.connect(user="scott", password="tiger", dsn="localhost/orclpdb")
    cursor = connection.cursor()
    cursor.execute(f"SELECT * FROM customers WHERE id = '{value}'")  # Noncompliant


def run_query_submodule_import(value):
    import oracledb.thin
    connection = oracledb.connect(user="scott", password="tiger", dsn="localhost/orclpdb")
    cursor = connection.cursor()
    cursor.execute(f"SELECT * FROM customers WHERE id = '{value}'")  # Noncompliant


def get_backup_connection():
    # Not an OracleDB connection at all (e.g. a MySQLdb connection), but this doesn't matter
    # once the file is OracleDB-tainted, see below.
    return mysqldb.connect(host="backup-host")


def run_query_unrelated_sink(value):
    # FP - see Archery/sql/engines/oracle.py:735-816, once isUsingOracleDB is set for the file
    # (because of oracledb.connect() elsewhere), *every* execute/executemany/parse/prepare/
    # fetch_df_all/fetch_df_batches call in the file is treated as an OracleDB sink, even when
    # the receiver clearly comes from an unrelated (non-OracleDB) connection/cursor object.
    connection = oracledb.connect(user="scott", password="tiger", dsn="localhost/orclpdb")
    backup_connection = get_backup_connection()
    backup_cursor = backup_connection.cursor()
    backup_cursor.execute(f"""insert into audit_log(msg) values('{value}')""")  # Noncompliant
