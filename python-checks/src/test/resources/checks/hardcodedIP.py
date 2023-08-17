"120"
ip = "1.2.3.4" # Noncompliant {{Make sure using this hardcoded IP address "1.2.3.4" is safe here.}}
#    ^^^^^^^^^
"1.2.3.4" # Noncompliant
"1.2.3.4:80" # Noncompliant
"1.2.3.4:8080" # Noncompliant
"1.2.3.4:a"
"1.2.3.4.5"

url = "http://192.168.0.1/admin.html"  # Noncompliant {{Make sure using this hardcoded IP address "192.168.0.1" is safe here.}}
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
url = "http://192.168.0.1:8181/admin.html" # Noncompliant
url = 'http://123.1.1.1' # Noncompliant
url = "http://www.example.org"

notAnIp1 = "0.0.0.1234"
notAnIp2 = "1234.0.0.0"
notAnIp3 = "1234.0.0.0.0.1234"
notAnIp4 = ".0.0.0.0"
notAnIp5 = "0.256.0.0"

fileName = "v0.0.1.200__do_something.sql" # Compliant - suffixed and prefixed
version = "1.0.0.0-1" # Compliant - suffixed

"1080:0:0:0:8:800:200C:417A" # Noncompliant {{Make sure using this hardcoded IP address "1080:0:0:0:8:800:200C:417A" is safe here.}}
"[1080::8:800:200C:417A]" # Noncompliant
"::800:200C:417A" # Noncompliant
"1080:800:200C::" # Noncompliant
"::FFFF:129.144.52.38" # Noncompliant
"::129.144.52.38" # Noncompliant
"::FFFF:38" # Noncompliant
"::100" # Noncompliant
"1080:0:0:0:8:200C:131.107.129.8" # Noncompliant
"1080:0:0::8:200C:131.107.129.8" # Noncompliant

"1080:0:0:0:8:800:200C:417G" # Compliant - not valid IPv6
"1080:0:0:0:8::800:200C:417A" # Compliant - not valid IPv6
"1080:0:0:0:8:::200C:417A" # Compliant - not valid IPv6
"1080:0:0:0:8" # Compliant - not valid IPv6
"1080:0::0:0:8::200C:417A" # Compliant - not valid IPv6
"1080:0:0:0:8::200C:417A:" # Compliant - not valid IPv6
"1080:0:0:0:8::200C:131.107.129.8" # Compliant - not valid IPv6
"1080:0:0:0:8::200C:256.256.129.8" # Compliant - not valid IPv6
"1080:0:0:0:8:200C:200C:131.107.129.8" # Compliant - not valid IPv6
"1080:0:0:0:8:131.107.129.8" # Compliant - not valid IPv6
"1080:0::0::8:200C:131.107.129.8" # Compliant - not valid IPv6
"1080:0:0:0:8:200C:131.107.129" # Compliant - not valid IPv6
"1080:0:0:0:8:200C:417A:131.107" # Compliant - not valid IPv6

url = "https://[3FFE:1A05:510:1111:0:5EFE:131.107.129.8]:8080/" # Noncompliant
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"https://[3FFE::1111:0:5EFE:131.107.129.8]:8080/" # Noncompliant

"ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff" # Noncompliant

# Exceptions
"0.0.0.0"
"::1"
"000:00::1"
'127.0.0.1'
"255.255.255.255"
"255.255.255.255:80"
"2.5.255.255"
"127.5.255.255"
"http://[::0]:100/"
"0000:0000:0000:0000:0000:0000:0000:0000"
''

# RFC 3849
"2001:db8:1f70::999:de8:7648:6e8"
"2001:db8:1f70::777:de8:ad2:6e8"
"http://[2001:db8:1f70::999:de8:7648:6e8]:100/"

# RFC 3849
"192.0.2.13"
"192.0.2.108"
"198.51.100.23"
"198.51.100.83"
"203.0.113.7"
"203.0.113.42"

'''http://123.1.1.1'''  # Noncompliant [[triple quoted but not multiline]]

'''123.1.1.1
some long message with ip address inside is compliant'''

r""" host=127.0.0.1 long strings with string prefix are also recognized as multiline strings and ignored by this rule"""

'time is 13:40:40'

passwd = 'root:aaa:16484:0:99999:7:::'  # OK, not an IP address (ref: SONARPY-196)

# SONARPY-270 should not raise issue on more than 4 IPV4 elements
SYS_OBJECT_ID = '1.3.6.1.2.1.1.2.0'


# local ipv4 mapped to ipv6 are compliants (ref: SONARPY-1058)
"::ffff:0:127.0.0.1"
"::ffff:0:127.82.255.17"
"::ffff:0:127.255.255.255"
"::FFFF:0:127.255.255.255"

"::ffff:127.0.0.1"
"::FFFF:127.0.0.1"
"::ffff:127.2.190.31"
"::ffff:127.255.255.255"
