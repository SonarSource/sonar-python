import flask
import mysql.connector
sql = flask.request.args.get('query')
conn = mysql.connector.connect(database='world')
cursor = conn.cursor()
cursor.execute(sql) # Noncompliant

x = mysql.connector
y = mysql.connector.connect
x
y
