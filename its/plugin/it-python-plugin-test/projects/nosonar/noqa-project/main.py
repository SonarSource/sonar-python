print "Hello"
print "Hello"  # noqa
print "Hello"  # noqa: PrintStatementUsage
print "Hello"  # noqa: PrintStatementUsage, S1309

# noqa with comments
print "Hello"  # noqa This is a comment
print "Hello"  # noqa: PrintStatementUsage This is a comment

# multiple issues
print "Hello"; print "World"
print "Hello"; print "World"  # noqa
print "Hello"; print "World"  # noqa: OneStatementPerLine,PrintStatementUsage
print "Hello"; print "World"  # noqa: OneStatementPerLine, PrintStatementUsage

# invalid noqa formats
print "Hello"  # noqa:
print "Hello"  # noqa: Invalid_Rule_Name
print "Hello"  # noqa PrintStatementUsage (missing colon)

# noqa at the end of file
print "Hello"  # noqa: PrintStatementUsage
