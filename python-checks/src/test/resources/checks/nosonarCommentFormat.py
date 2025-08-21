
# NOSONAR
# NOSONAR()
# NOSONAR(a, b)
# NOSONAR with some text
# NOSONAR() with some text
# NOSONAR(a, b) with some text
# NOSONAR ()
# NOSONARa
# noqa
# noqa: a,b
# noqa: a, b

# Noncompliant@+1
# NOSONAR(
# Noncompliant@+1
# NOSONAR)
# Noncompliant@+1
# NOSONAR)(
# Noncompliant@+1
# NOSONAR(,)
# Noncompliant@+1
# NOSONAR(a,)
# Noncompliant@+1
# NOSONAR(a (b))
# Noncompliant@+1
# noqa: a,,c
# Noncompliant@+1
# noqa: a,c,

# noqa: a, some text

# Noncompliant@+1
# noqa: F321 NOSONAR

# noqa: F321 # NOSONAR

# NOSONAR(S1234)
# NOSONAR(S1)
# NOSONAR(NoSonar)
# Noncompliant@+1
# NOSONAR(python:S1234)
