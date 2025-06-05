data = [3, 1, 4, 1, 5, 9]
result = set(sorted(data)) # Noncompliant {{Remove either the call to set or sorted.}}
#        ^^^

sorted_data = sorted(data)
#             ^^^^^^> {{The list is sorted here.}}
assigned = set(sorted_data) # Noncompliant
#          ^^^
just_set = set(data) 

unique_sorted = sorted(set(data)) 

# ============== COVERAGE ================

a = list(sorted(data))
a = set([1,2])

multiple_assignement = sorted(data)
multiple_assignement = [1,3,3]
b = set(multiple_assignement)

unsorted = list(data)
c = set(unsorted)

from module import something
assignment = something[0:1]
d = set(assignement)
