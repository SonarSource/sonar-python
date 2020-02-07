
CORS_ORIGIN_ALLOW_ALL = True # Noncompliant {{Make sure this permissive CORS policy is safe here.}}
CORS_ORIGIN_REGEX_WHITELIST = [r".*"] # Noncompliant
CORS_ORIGIN_REGEX_WHITELIST = [r"^.*$"] # Noncompliant
CORS_ORIGIN_REGEX_WHITELIST = [r".+"] # Noncompliant
CORS_ORIGIN_REGEX_WHITELIST = [r"^.+$"] # Noncompliant

CORS_ORIGIN_REGEX_WHITELIST = [42] # Compliant
CORS_ORIGIN_ALLOW_ALL = False # Compliant
CORS_ORIGIN_ALLOW_ALL = "True" # Compliant
CORS_ORIGIN_WHITELIST = ["trustedwebsite.com"] # Compliant
CORS_URLS_REGEX = r'^.*$' # Compliant

