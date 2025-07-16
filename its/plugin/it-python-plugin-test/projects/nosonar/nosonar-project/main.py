print "Hello"
print "Hello" # NOSONAR
print "Hello" # NOSONAR(PrintStatementUsage)
print "Hello" # NOSONAR(PrintStatementUsage, NOSONAR)

print "Hello" # NOSONAR This is a comment
print "Hello" # NOSONAR(PrintStatementUsage,) This is a comment

print "Helo"; print "hello"
print "Helo"; print "hello" # NOSONAR
print "Helo"; print "hello" # NOSONAR(PrintStatementUsage)

# Wrongly formatted NOSONAR
print "Hello" # NOSONAR(
print "Hello" # NOSONAR(PrintStatementUsage, NOSONAR
print "Hello" # NOSONAR(PrintStatementUsage  NOSONAR)


# NOSONAR on the last line
print "Helo"; print "hello" # NOSONAR(PrintStatementUsage, OneStatementPerLine)
