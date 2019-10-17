#!/usr/bin/python

# Several other operations on lists.

fred = [ 'Alice', 'goes', 'to', 'market' ]
print 'A:', fred

fred.extend([ 'with', 'Mike' ])
print 'B:', fred

last = fred.pop()
fred.append('Fred')

print 'C:', fred

print 'So much for Mike.'
print 'There are', len(fred), 'items in fred.'
print 'The word market is located at position', fred.index('market')

fred = [ 'On', 'Tuesday,' ] + fred
print 'D:', fred

fred.reverse()
print 'E:', fred

fred.sort()
print 'F:', fred
