
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

# nosec
# nosec B101
# nosec B101, B102
# nosec B101, B102 reason text
# nosec: S1234, S5678
# nosec because of foo, bar, and baz
# Noncompliant@+1
# nosec B101,
# Noncompliant@+1
# nosec ,B101
# Noncompliant@+1
# nosec B101,,B102
# Noncompliant@+1
# nosec B101 reason, B102
