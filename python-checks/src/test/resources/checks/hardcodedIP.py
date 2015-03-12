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