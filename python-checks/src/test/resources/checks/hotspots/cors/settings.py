
CORS_ORIGIN_ALLOW_ALL = True # Noncompliant {{Make sure this permissive CORS policy is safe here.}}
CORS_ORIGIN_REGEX_WHITELIST = [r".*"] # Noncompliant
CORS_ORIGIN_REGEX_WHITELIST = [r"^.*$"] # Noncompliant
CORS_ORIGIN_REGEX_WHITELIST = [r".+"] # Noncompliant
CORS_ORIGIN_REGEX_WHITELIST = [r"^.+$"] # Noncompliant

CORS_ORIGIN_ALLOW_ALL = False # Compliant
CORS_ORIGIN_WHITELIST = ["trustedwebsite.com"] # Compliant
CORS_URLS_REGEX = r'^.*$' # Compliant
CORS_ALLOW_METHODS =  [ # Compliant
    'DELETE',
    'GET',
    'OPTIONS',
    'PATCH',
    'POST',
    'PUT',
]
CORS_ALLOW_HEADERS = [] # Compliant
CORS_EXPOSE_HEADERS = [] # Compliant
CORS_PREFLIGHT_MAX_AGE = 86400 # Compliant
CORS_ALLOW_CREDENTIALS = True # Compliant
