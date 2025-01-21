import flask

app1 = flask.Flask("app1")
app2 = flask.app.Flask("app2")

app1 # known flask.app.Flask (v1)
app2 # unknown (v1) - both are known in v2
