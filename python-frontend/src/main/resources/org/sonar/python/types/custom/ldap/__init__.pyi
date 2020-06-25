from .ldapobject import SimpleLDAPObject as SimpleLDAPObject
from .ldapobject import ReconnectLDAPObject as ReconnectLDAPObject
from .ldapobject import LDAPObject as LDAPObject

def initialize(*args, **kwargs) -> LDAPObject: ...
