import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('127.0.0.1', 50007))  # Noncompliant

address = 'http://123.1.1.1' # Noncompliant

address2 = 'https://255.1.1.1/some/path' # Noncompliant

str = "123.1111.1.1 String without ip "

'''http://123.1.1.1'''  # multiline strings are not considered


message = '''123.1.1.1
some long message with ip address inside is compliant'''

FASTCGI_HELP = r""" host=127.0.0.1 long strings with string prefix are also recognized as multiline strings and ignored by this rule"""

str = '2001:0db8:11a3:09d7:1f34:8a2e:07a0:765d'
str = '::1f34:8a2e:07a0:765d'
str = '1f34:8a2e:07a0:765d::'
str = '1f34:2e:7a0:765d::'
str = 'time is 13:40:40'

str = '1.2.3.4'
