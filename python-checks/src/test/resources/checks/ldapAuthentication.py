import ldap
import os

def fn(p):
    connect = ldap.initialize('ldap://127.0.0.1:1389')
    connect.set_option(ldap.OPT_REFERRALS, 0)

    connect.simple_bind('cn=root') # Noncompliant {{Provide a password when authenticating to this LDAP server.}}
#   ^^^^^^^^^^^^^^^^^^^
    connect.simple_bind_s('cn=root') # Noncompliant
    connect.bind_s('cn=root', None) # Noncompliant
#   ^^^^^^^^^^^^^^            ^^^^<
    connect.bind('cn=root', None) # Noncompliant

    connect.bind('cn=root', "") # Noncompliant
    connect.bind('cn=root', cred="") # Noncompliant
    connect.bind('cn=root', serverctrls=os.environ.get('LDAP_PASSWORD')) # Noncompliant

    pwd = ""
#         ^^>
    connect.bind('cn=root', pwd) # Noncompliant
#   ^^^^^^^^^^^^            ^^^<

    connect.simple_bind() # Noncompliant

    args = ['cn=root']
    connect.simple_bind(*args) # FN


def compliant(p):
    connect = ldap.initialize('ldap://127.0.0.1:1389')
    connect.set_option(ldap.OPT_REFERRALS, 0)

    connect.simple_bind('cn=root', os.environ.get('LDAP_PASSWORD'))
    connect.simple_bind_s('cn=root', os.environ.get('LDAP_PASSWORD'))
    connect.bind_s('cn=root', os.environ.get('LDAP_PASSWORD'))
    connect.bind('cn=root', os.environ.get('LDAP_PASSWORD'))

    connect.bind('cn=root', "foobar")
    connect.bind('cn=root', cred=os.environ.get('LDAP_PASSWORD'))

    args = ['cn=root', os.environ.get('LDAP_PASSWORD')]
    connect.simple_bind(*args)

    pwd = os.environ.get('LDAP_PASSWORD')
    connect.bind('cn=root', pwd)

    if p:
        ambiguous_pwd = ""
    else:
        ambiguous_pwd = "foo"
    connect.bind('cn=root', ambiguous_pwd)

    unknown_connect.bind('cn=root', os.environ.get('LDAP_PASSWORD'))
