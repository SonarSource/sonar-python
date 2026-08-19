# Noncompliant@+1 {{Rename this parameter "inputPar2" to match the regular expression ^[_a-z][a-z0-9_]+$.}}
def fun(input_par1, inputPar2, inputPar3 = 3): # Noncompliant {{Rename this parameter "inputPar3" to match the regular expression ^[_a-z][a-z0-9_]+$.}}
    someName = 1 # Noncompliant {{Rename this local variable "someName" to match the regular expression ^[_a-z][a-z0-9_]+$.}}
#   ^^^^^^^^
    CamelName: int = 1 # Noncompliant {{Rename this local variable "CamelName" to match the regular expression ^[_a-z][a-z0-9_]+$.}}
#   ^^^^^^^^^
    another_name = someName
    someName = 3
    inputPar2 = 2



def fun2():
    x = 1 # Noncompliant
    for i in range(3):
        print(x)
        x = i
# Noncompliant@+1
def fun3((inputPar2, input_par3), inputPar1=global_var): # Noncompliant
    pass

def fun4():
    x = 1 # Noncompliant
    def fun5():
        # Noncompliant@+2
        # Noncompliant@+1
        x = 2; y = 2
        print(1)

def fun6():
    # Noncompliant@+3
    # Noncompliant@+2
    # Noncompliant@+1
    (a, (b, c)) = (1, (2, 3))

    # Noncompliant@+2
    # Noncompliant@+1
    d, e = 1, 2

# Noncompliant@+2
# Noncompliant@+1
def fun7(*ID, **ID2):
    ID += (2,)
    print(ID)
    print(ID2)


def fun8():
    x = MyClass() # Noncompliant
    x.b = 2
    x = [1]
    for i in range(1):
        x[i] = 3

def fun9():
    CONSTANT_NAME = "Hello, world"
    for counterName in names: # Noncompliant {{Rename this local variable "counterName" to match the regular expression ^[_a-z][a-z0-9_]+$.}}
        pass

def fun10():
    # Noncompliant@+2
    # Noncompliant@+1
    a = b = 1
    name = d.e = 1
    #Noncompliant@+1
    for (builderName, stepName) in foo: #Noncompliant
      pass


def type_variables_ok(MyTypeParameter: type):
    from collections import namedtuple
    Person = namedtuple('Person', ['name', 'age', 'gender'])

    from typing import NamedTuple
    Employee = NamedTuple('Employee', [('name', str), ('id', int)])

def type_aliases_from_typing_special_form():
    from typing import Type, Union
    class A: ...
    MyType = Type[A] # OK
    AliasType = Union[str, int] # OK

    # Assigned from a subscription check, but not from a typing_SpecialForm
    array = [1,2,3]
    MyInt = array[1]  # Noncompliant
    MyOtherInt = [1,2,3][1] # Noncompliant

def type_variables_fp():
    class MyClass:
        ...
    MyClassAlias = MyClass  # OK
    MyTypeVariable: type = unknown_call()  # Noncompliant

def builtin_type_assignments(role, roles):
    RoleType = type(role)
    OptionalRoleType = type(roles[0]) if roles else None
    ParenthesizedRoleType = (type(role))
    BadName = wrapper(type(role))  # Noncompliant

def annotated_builtin_type_assignments(role, roles):
    AnnotatedRoleType: type = type(role)
    AnnotatedOptionalRoleType: type | None = type(roles[0]) if roles else None
    AnnotatedBadName: type = wrapper(type(role))  # Noncompliant

def walrus_builtin_type_assignments(role):
    if (WalrusRoleType := type(role)):
        pass
    while (WalrusBadName := wrapper(type(role))):  # Noncompliant
        pass

def conditional_builtin_type_assignments(role, fallback, cond, roles):
    BothBranchesType = type(role) if cond else type(fallback)
    NestedType = type(role) if cond else type(fallback) if roles else None
    ReversedType = None if not roles else type(roles[0])
    ParenthesizedNoneType = type(role) if cond else (None)
    BothBranchesNone = None if cond else None  # Noncompliant
    MixedBranches = type(role) if cond else 5  # Noncompliant

def reassigned_parameter(BadParam, role):  # Noncompliant
    BadParam = type(role)

def shadowed_type_assignment():
    def type(value):
        return value

    BadName = type("role")  # Noncompliant

def ml_names():
    X = [1, 2, 3]
    Y = [0, 1, 0]
    X_train = X
    Y_train = Y
    X_test = X
    Y_test = Y

    return X_train, Y_train, X_test, Y_test

def django_models():
    from django.apps import AppConfig

    class RockNRollConfig(AppConfig):
        def ready(self):
            MyModel = self.get_model("MyModel") 

    MySecondModel = RockNRollConfig().get_model("MySecondModel")  

    from django.apps import apps
    Product = apps.get_model('shop', 'Product')


import json
from typing import Any

class CustomJSONEncoder(json.JSONEncoder):

    def default(self, o: Any) -> dict[str, Any]:  # Compliant: `o` parameter name comes from json.JSONEncoder.default
        return {"value": str(o)}


class EdgeCase:
    method = None
    
    def method(self, ID):  # Noncompliant
        pass


class BaseClass:
    def process(self, data_input):
        pass

class DerivedClass(BaseClass):
    def process(self, ID):  # Noncompliant
        pass

class BaseShortSignature:
    def execute(self):
        pass

class DerivedLongerSignature(BaseShortSignature):
    def execute(self, ExtraParam):  # Noncompliant
        pass


import marimo

app = marimo.App()


@app.cell
def _():
    TSHIRT_ORDER = ["XS", "S", "M", "L", "XL", "XXL"]
    return TSHIRT_ORDER,


@app.cell
def _(TSHIRT_ORDER):
    filtered = [size for size in TSHIRT_ORDER if size != "XXL"]
    return filtered,


@app.cell(hide_code=True)
def _(API_KEY, DB_CONFIG):
    connection_string = f"host={DB_CONFIG['host']}?key={API_KEY}"
    return connection_string,


@app.function
def helper(INPUT_DATA):
    return INPUT_DATA
