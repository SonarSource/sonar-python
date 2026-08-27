from oracledb import connect


def run_query(value):
    connection = connect(user="scott", password="tiger", dsn="localhost/orclpdb")
    cursor = connection.cursor()
    cursor.execute(f"SELECT * FROM customers WHERE id = '{value}'")  # Noncompliant {{Make sure that formatting this SQL query is safe here.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    cursor.execute("SELECT * FROM customers WHERE id = :id")  # OK


def run_query_cx_oracle(value):
    from cx_Oracle import connect as cx_connect
    connection = cx_connect(user="scott", password="tiger", dsn="localhost/orclpdb")
    cursor = connection.cursor()
    cursor.execute(f"SELECT * FROM customers WHERE id = '{value}'")  # Noncompliant
