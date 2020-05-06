MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
#    'django.middleware.csrf.CsrfViewMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
] # Noncompliant {{Make sure disabling CSRF protection is safe here.}}

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
] # Compliant


# Corner cases for code coverage
MIDDLEWARE = [
  'it.s.called.middleware.but.content.doesnt.start.with.expected.prefix',
  f'does this have more than one part {"yes"} hm, no apparently not',
  ""
]

MIDDLEWARE[42] = 'not an array'

that_s_some_completely_unrelated_array = [
  'django.stuff.it.contains',
  'django.but.it.shouldn.t.matter'
] # Compliant
