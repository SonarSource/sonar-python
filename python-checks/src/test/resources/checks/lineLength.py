#Noncompliant@+1
#Noncompliant@+1 {{The line contains 40 characters which is greater than 30 authorized.}}
print "This line contains 40 characters"
print
print """This is a multiline
string literal."""

# Noncompliant@+1
from some.where import foo, bar

# Noncompliant@+1
something = call(('arg1', 'arg2',
# Noncompliant@+1
                  'arg3', 'ar'),
                 'arg'
                 )

something_esle = [
    ("hello", {},
     # Noncompliant@+1
     u'hello hello hello hello'),

    ("a", {},
     u'b'),
]

another = Call(
    foo("something",
        # Noncompliant@+1
        bar("more than 30 ch"),
        ),

    foo("something",
        bar("ok"),
        ),
)
