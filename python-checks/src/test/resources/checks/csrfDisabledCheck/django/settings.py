MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
#    'django.middleware.csrf.CsrfViewMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
] # Noncompliant {{CSRF protection (django.middleware.csrf.CsrfViewMiddleware) is missing.}}

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
] # Compliant
