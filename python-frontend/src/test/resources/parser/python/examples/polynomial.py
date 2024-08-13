#
# This module contains operations to manipulate polynomials.
#

# Need some string services, and some standard system services.
import string, sys

#
# Function to evaluate a polynomial at x.  The polynomial is given
# as a list of coefficients, from the greatest to the least.  It returns
# the value of the polynomial at x.
def eval(x, poly):
    '''Evaluate at x the polynomial with coefficients given in poly.
    The value p(x) is returned.'''

    sum = 0
    while 1:
        sum = sum + poly[0]	# Add the next coef.
        poly = poly[1:]		# Done with that one.
        if not poly: break	# If no more, done entirely.
        sum = sum * x		# Mult by x (each coef gets x right num times)

    return sum

#
# Function to read a line containing a list of integers and return
# them as a list of integers.  If the string conversion fails, it
# returns the empty list.  The input 'quit' is special.  If that is
# this input, this value is returned.  The input comes from the file
# if given, otherwise from standard input.  If the prompt is given, it
# is printed first.  If reading reaches EOF, or the input line quit,
# the call throws EOFError.  If the conversion fails, it throws
# ValueError.
def read(prompt = '', file = sys.stdin):
    '''Read a line of integers and return the list of integers.'''

    # Read a line
    if prompt: print prompt,
    line = file.readline()
    if not line:
        raise EOFError, 'File ended on attempt to read polynomial.'
    line = line[:-1]
    if line == 'quit':
        raise EOFError, 'Input quit on attempt to read polynomial.'

    # Go through each item on the line, converting each one and adding it
    # to retval.
    retval = [ ];
    for str in string.split(line):
        retval.append(int(str))

    return retval

#
# Create a string of the polynomial in sort-of-readable form.
def srep(p):
    '''Print the coefficient list as a polynomial.'''

    # Get the exponent of first coefficient, plus 1.
    exp = len(p)

    # Go through the coefs and turn them into terms.
    retval = ''
    while p:
        # Adjust exponent.  Done here so continue will run it.
        exp = exp - 1

        # Strip first coefficient
        coef = p[0]
        p = p[1:]

        # If zero, leave it out.
        if coef == 0: continue

        # If adding, need a + or -.
        if retval:
            if coef >= 0:
                retval = retval + ' + '
            else:
                coef = -coef
                retval = retval + ' - '

        # Add the coefficient, if needed.
        if coef != 1 or exp == 0:
            retval = retval + str(coef)
            if exp != 0: retval = retval + '*'

        # Put the x, if we need it.
        if exp != 0:
            retval = retval + 'x'
            if exp != 1: retval = retval + '^' + str(exp)

    # For zero, say that.
    if not retval: retval = '0'

    return retval
