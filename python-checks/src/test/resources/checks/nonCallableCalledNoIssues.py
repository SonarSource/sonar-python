
def param_with_type_hint(a: int):
    a() # OK

def call_unknown_value(unknown_value):
    (~unknown_value)(X) # Ok

import socket
def type_alias_is_callable():
    # socket.error is a TypeAlias for OSError in typeshed — calling it must not trigger S5756
    e = socket.error(42, "connection refused")  # OK


# Model.delete is declared as "classmethod_only[...]": reading it returns the result of classmethod_only.__get__
import peewee

class KeyValue(peewee.Model):
    key = peewee.CharField(primary_key=True)

def class_level_descriptor_attribute(key):
    KeyValue.delete().where(KeyValue.key == key).execute()
