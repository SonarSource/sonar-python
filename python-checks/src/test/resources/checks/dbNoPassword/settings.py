import os

PASSWORD = "password"

DATABASES = {
    'postgresql_db': {
        'PASSWORD': '',                            # Noncompliant {{Add password protection to this database.}}
#       ^^^^^^^^^^^^^^
        'PASSWORD': os.getenv('DB_PASSWORD'),      # Compliant
        'HOST': '',                                # Compliant
        PASSWORD: '',                                # Compliant
    },
    'mysql_db': {
        "PASSWORD": "",                          # Noncompliant
    },
    'oracle_db': {
        'PASSWORD': '',                           # Noncompliant
    },
    'other_key': {
        'PASSWORD': '',                            # Noncompliant
        'PASSWORD': os.getenv('DB_PASSWORD'),      # Compliant
    },
}

OTHER = {
    'postgresql_db': {
        'PASSWORD': '',                            # Compliant
    }
}

DATABASES[42] = {
    'postgresql_db': {
        'PASSWORD': '',                            # Compliant
    }
}

DATABASES, OTHER = {
    'postgresql_db': {
        'PASSWORD': '',                            # out of scope
    }
}, 42

DATABASES = {
    'postgresql_db': {
        'PASSWORD': '',                            # Noncompliant
    },
    'mysql_db': {
        'PASSWORD': os.getenv('DB_PASSWORD'),
    },
}
