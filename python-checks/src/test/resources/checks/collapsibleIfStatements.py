if a:
  if b:
    pass
  if c:
    pass

if a:
  if b:
    pass
  else:
    pass

if a:
  if b: # NOK
    pass

if a:
  pass
elif b:
  if c: # NOK
    pass

if a:
  pass
elif b:
  if c:
    pass
else:
  pass

if a:
  if b: # NOK
    if c: # NOK
      pass

if a:
  if b: # NOK
    if c:
      pass
    else:
      pass

if a:
  if b:
    pass
  elif c:
    pass

if a:
  if b:
    pass
  do_something()
