#!/usr/bin/python

#
# This program evaluates polynomials.  It first asks for the coefficients
# of a polynomial, which must be entered on one line, highest-order first.
# It then requests values of x and will compute the value of the poly for
# each x.  It will repeatly ask for x values, unless you the user enters
# a blank line.  It that case, it will ask for another polynomial.  If the
# user types quit for either input, the program immediately exits.
#

# Need some string services, and some standard system services.
import string, polynomial

#
# Run until some kind of endfile.
try:
    # Repeat until an exception or quit gets us out.
    while 1:
        # Read a poly until it works.  An EOF will except out of the
        # program.
        while 1:
            try:
                poly = polynomial.read('Enter a polynomial coefficients: ')
            except:
                print 'Conversion failed.  Please try again.'
            else:
                break

        # Read and evaluate x values until the user types a blank line.
        # Again, and EOF will except out of the pgm.
        while 1:
            resp = raw_input('Enter x value or blank line: ')
            if resp == 'quit': raise EOFError
            if not resp: break
            try:
                x = int(resp)
            except ValueError:
                print "That doesn't look like an integer.  Please try again."
            else:
                print 'p(x) =', polynomial.srep(poly)
                print 'p(' + str(x) + ') =', polynomial.eval(x, poly)

except (EOFError, KeyboardInterrupt):
    # Exit without error for EOF or ^C.  Print a blank line to clear after
    # any prompt.
    print
