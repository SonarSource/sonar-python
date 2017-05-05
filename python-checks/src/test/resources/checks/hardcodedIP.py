import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('127.0.0.1', 50007))  # Noncompliant {{Make this IP "127.0.0.1" address configurable.}}
#       ^^^^^^^^^^^

address = 'http://123.1.1.1' # Noncompliant

address2 = 'https://255.1.1.1/some/path' # Noncompliant
#          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

str = "String 123.111.1.1 is IPv4"       # Noncompliant
str = "String 123.0111.1.1 is not IPv4"
str = "String 123.111.1.1000 is not IPv4"
str = "String 0123.111.1.1 is not IPv4"
str = "String 294.125.1.1 is not IPv4"

'''http://123.1.1.1'''  # multiline strings are not considered

message = '''123.1.1.1
some long message with ip address inside is compliant'''

FASTCGI_HELP = r""" host=127.0.0.1 long strings with string prefix are also recognized as multiline strings and ignored by this rule"""

str = '2001:0db8:11a3:09d7:1f34:8a2e:07a0:765d' # Noncompliant
str = '::1f34:8a2e:07a0:765d' # Noncompliant
str = '1f34:8a2e:07a0:765d::' # Noncompliant
str = '1f34:2e:7a0:765d::'    # Noncompliant {{Make this IP "1f34:2e:7a0:765d::" address configurable.}}
str = '1f34:2e:7a0:765d0::'   # not an IPv6, as one token contains 5 characters
str = '1f34:2e:7a0::765dA'    # not an IPv6, as one token contains 5 characters
str = '01f34:2e:7a0:765d::'   # not an IPv6, as one token contains 5 characters

str = 'time is 13:40:40'
str = '1.2.3.4'               # Noncompliant
str = 'http://[1080:0:0:0:8:800:200C:417A]:8888/index.html'    # Noncompliant {{Make this IP "1080:0:0:0:8:800:200C:417A" address configurable.}}
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

passwd = 'root:aaa:16484:0:99999:7:::'  # OK, not an IP address (ref: SONARPY-196)
