if foo == 'blah': do_blah_thing() #Noncompliant
do_one(); do_two(); do_three() #Noncompliant{{At most one statement is allowed per line, but 3 statements were found on this line.}}

if foo == 'blah':
    do_blah_thing()
do_one()
do_two()
do_three()
