import ssl

# Bare nosec suppresses every rule on the line (Bandit semantics).
ssl.SSLContext(ssl.PROTOCOL_TLSv1)
ssl.SSLContext(ssl.PROTOCOL_TLSv1)  # nosec

# Scoped nosec with a Bandit ID narrows suppression to that ID; B502 has no Sonar mapping, so S4423 still fires.
ssl.SSLContext(ssl.PROTOCOL_TLSv1)  # nosec B502 legacy peer

# Selective suppression by Sonar rule key.
ssl.SSLContext(ssl.PROTOCOL_TLSv1)  # nosec S4423

# Selective suppression narrows to the listed keys; OneStatementPerLine still fires.
ssl.SSLContext(ssl.PROTOCOL_TLSv1); ssl.SSLContext(ssl.PROTOCOL_TLSv1_1)  # nosec S4423

# Baseline: two S4423 + OneStatementPerLine all fire.
ssl.SSLContext(ssl.PROTOCOL_TLSv1); ssl.SSLContext(ssl.PROTOCOL_TLSv1_1)

# Bare nosec suppresses security rules but lets non-security rules (e.g. OneStatementPerLine) through.
ssl.SSLContext(ssl.PROTOCOL_TLSv1); ssl.SSLContext(ssl.PROTOCOL_TLSv1_1)  # nosec

# Trailing directive on the last statement.
ssl.SSLContext(ssl.PROTOCOL_TLSv1)  # nosec

# Selective suppression lets S4423 through
ssl.SSLContext(ssl.PROTOCOL_TLSv1)  # nosec S1234
