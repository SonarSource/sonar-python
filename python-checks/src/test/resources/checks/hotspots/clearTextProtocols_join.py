# SONARPY-4809: bare schemes at the start of a collection passed to an empty-string join.
def inline_parts(host):
    target = "".join(["http://", host]) # Noncompliant {{Using HTTP protocol is insecure. Use HTTPS instead.}}
    #                 ^^^^^^^^^
    target = "".join(("ftp://", host)) # Noncompliant
    target = "".join(["https://", host]) # Compliant
    # The minimal check does not inspect how many other parts exist or their values.
    target = "".join(["http://"]) # Noncompliant
    target = "".join(["http://", "localhost"]) # Noncompliant
    descriptions = ", ".join(["http://", "ftp://"]) # Compliant
    descriptions = str(["http://", "ftp://"]) # Compliant: this is not a join.
    target = "".join(["prefix", "http://"]) # Compliant: the scheme is not the first part.


def assigned_parts(host):
    parts = ["http://", host] # Noncompliant
    return "".join(parts)


def appended_parts(host):
    parts = ["http://"] # Noncompliant
    parts.append(host)
    return "".join(parts)


def unused_parts(host):
    parts = ["http://"] # Compliant: the list is never joined.
    parts.append(host)


# Ruling: salt/returners/splunk.py:152. The initializer is conditional and parts are appended in a loop.
def collector_url(server, port, tls):
    if tls:
        parts = ["https://"]
    else:
        parts = ["http://"] # Noncompliant
    for part in [server, ":", port, "/services/collector/event"]:
        parts.append(part)
    return "".join(parts)
