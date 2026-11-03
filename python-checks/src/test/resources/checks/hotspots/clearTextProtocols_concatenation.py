# SONARPY-5033: adjacent literals and explicit + construction.
def adjacent_literals(url: str):
    target = "http://" "service.company.com" # Noncompliant {{Using HTTP protocol is insecure. Use HTTPS instead.}}
    #        ^^^^^^^^^
    target = ("ftp://" "files.company.com") # Noncompliant
    # Accepted trade-off: adjacent elements are counted without evaluating their contents.
    target = "http://" "localhost" # Noncompliant
    target = "http://" "example.com" # Noncompliant
    target = "http://" "" # Noncompliant
    target = "http://localhost" "/path" # Compliant - the first element already includes localhost, so the existing safe-host exemption applies.
    target = "https://" "service.company.com" # Compliant
    target = "description: " "http://service.company.com" # Noncompliant
    url.startswith("http://" "service.company.com") # Compliant
    url.startswith("http://" "localhost") # Compliant
    url.replace("http://" "service.company.com", "https://service.company.com") # Compliant
    url.replace("https://", "http://" "") # Noncompliant


# Check every bare scheme in an adjacent-literal group.
schemes = (
    "http://" # Noncompliant
    "ftp://" # Noncompliant
)


# A safe first URL must not hide a complete unsafe URL in a later literal element.
origins = (
    "http://localhost:3000,"
    "http://api.company.com" # Noncompliant {{Using HTTP protocol is insecure. Use HTTPS instead.}}
)

# Keep a separate finding for each complete unsafe URL element.
origins = (
    "http://api.company.com," # Noncompliant {{Using HTTP protocol is insecure. Use HTTPS instead.}}
    "ftp://files.company.com" # Noncompliant {{Using FTP protocol is insecure. Use SFTP, SCP or FTPS instead.}}
)


def direct_concatenation(host, path, url: str):
    target = "http://" + "service.company.com" # Noncompliant
    # Ruling: nltk/downloader.py:1276 and the two Tornado escape.py copies (326 and 282).
    target = "http://" + host # Noncompliant {{Using HTTP protocol is insecure. Use HTTPS instead.}}
    #        ^^^^^^^^^
    # Ruling: salt/modules/zabbix.py:121 and tornado/web.py:1375.
    target = "http://" + host + "/zabbix/api_jsonrpc.php" # Noncompliant
    target = "http://" + host + "/" # Noncompliant
    target = (("ftp://")) + host # Noncompliant
    target = "ws://" + host + "/socket" # Noncompliant
    target = "http://service.company.com" + path # Noncompliant
    target = "https://" + host # Compliant
    # Accepted trade-off: + does not evaluate the other operands.
    target = "http://" + "localhost" # Noncompliant
    target = "http://" + "example.com" # Noncompliant
    target = "http://" + "" # Noncompliant
    # A complete safe literal keeps the shared-filter exemption (Indico regression).
    target = "http://localhost" + path.format(123) # Compliant
    url.startswith("http://" + host + "/") # Compliant
    url.startswith(("http://" + host, "ftp://" + host)) # Compliant
    str.startswith(url, ("http://") + host) # Compliant
    url.replace("http://" + host, "https://" + host) # Compliant
    url.removeprefix("http://" + host) # Compliant
    url.removesuffix("ftp://" + host) # Compliant
    url.replace("https://", "http://" + host) # Noncompliant
    str.replace(url, "https://", "http://" + host) # Noncompliant


def formatted_url(host, port):
    # Preserve one report for this existing f-string case.
    target = f"http://{host}:{port}/" # Noncompliant
