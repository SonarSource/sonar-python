import pgdb
sql = flask.request.args.get('query')
conn = pgdb.connect()
cursor = conn.cursor()
cursor.execute(sql) # Noncompliant
cursor.executemany(sql) # Noncompliant

x = pgdb.connect
x
