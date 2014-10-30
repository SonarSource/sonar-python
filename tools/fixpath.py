import re
import sys

PATTERN = 'classname="(.*)"'


def replfunc(match):
    global path_to_prepend
    return 'classname="%s%s"' % (path_to_prepend, match.group(1))


def usage():
    return "Usage: %s <infile> <outfile> <path to prepend>" % sys.argv[0]


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print usage()
        sys.exit(-1)

    infile = sys.argv[1]
    outfile = sys.argv[2]
    path_to_prepend = sys.argv[3]

    with open(infile, "r") as infd:
        with open(outfile, "w") as outfd:
            for line in infd:
                outfd.write(re.sub(PATTERN, replfunc, line))
