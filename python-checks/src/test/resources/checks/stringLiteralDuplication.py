print("aa a") # ok, to short
print("aa a")
print("aa a")

print("aa aa") # Noncompliant {{Define a constant instead of duplicating this literal "aa aa" 3 times.}} [[effortToFix=2]]
#     ^^^^^^^
print("aa aa")
#     ^^^^^^^<
print("aa aa")
#     ^^^^^^^<

print("bb bb") # ok, only 2 literals
print("bb bb")

print(r"bb\\bb") # Noncompliant
print(r"bb\\bb")
print(r"bb\\bb")

print("aa\\aa") # compliant
print("aa\\aa")
print(r"aa\\aa")

name = "Alice"
print(f"hello {name}")
print(f"hello {name}")
name = "Bob"
print(f"hello {name}")

from flask import Flask
app = Flask(__name__)

@app.route("/api/users/", methods=['GET'])
def users_get():
    pass

@app.route("/api/users/", methods=['PUT'])
def users_put():
    pass

@app.route("/api/users/", methods=['POST'])
def users_post():
    pass

options = { 'suppress': False }
if options['suppress']:
    print(options['suppress'])
