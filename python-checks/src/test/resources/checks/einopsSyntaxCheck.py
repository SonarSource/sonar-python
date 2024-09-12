import test
import torch
import einops
from einops import reduce, repeat, rearrange

img = torch.randn(32, 32, 3)

rearrange(img, 'h w c -> c h w')

imgs = torch.randn(10, 32, 32, 3)

rearrange(imgs, 'b h w c -> b c h w')
rearrange(imgs, 'b h w c -> (b h) w c')

unstacked = rearrange(imgs, '(b h) w c -> b h w c', b=10)

unstacked2 = rearrange(imgs, '(... h) w c -> ... h w c') # Noncompliant {{Fix the syntax of this einops operation: Ellipsis inside parenthesis on the left side is not allowed.}}
                            #^^^^^^^^^^^^^^^^^^^^^^^^^^
unstacked2 = einops.rearrange(imgs, '(... h) w c -> ... h w c') # Noncompliant 
                                   #^^^^^^^^^^^^^^^^^^^^^^^^^^
unstacked2 = einops.rearrange(pattern='(... h) w c -> ... h w c', tensor=imgs) # Noncompliant 
                                     #^^^^^^^^^^^^^^^^^^^^^^^^^^
unstacked2 = rearrange(imgs, '(... h) w c -> (... h) w c') # Noncompliant
                            #^^^^^^^^^^^^^^^^^^^^^^^^^^^^
unstacked2 = rearrange(imgs, 'h w c -> (... h) w c')
test.rearrange(imgs, '(...) -> (... h) w c ') 

repeat(imgs, "(h) (( w c) -> (h w c)") # Noncompliant {{Fix the syntax of this einops operation: nested parenthesis are not allowed.}}
repeat(imgs, "(h w c -> (h w c)") # Noncompliant {{Fix the syntax of this einops operation: parenthesis are unbalanced.}}
repeat(imgs, "h w c -> h w c))") # Noncompliant 
repeat(imgs, "h w c -> h w c(") # Noncompliant 
repeat(imgs, "h w c) -> h w c") # Noncompliant 
repeat(imgs, "h w c -> (h w c(") # Noncompliant 

rearrange(imgs, ")h w c -> h w c") # Noncompliant 


reduce(imgs, 'b c -> b c', 'max')
rearrange(imgs, "h w c -> h w c", 1) # Not a correct parameter but still we should not raise.
reduce(imgs, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h2=2, w2=2) 
rearrange(imgs, 'b c -> b c', h2=2, w2=2) # Noncompliant {{Fix the syntax of this einops operation: the parameters h2, w2 do not appear in the pattern.}}
rearrange(imgs, "(b h) w c -> b h w c ", b1=1) # Noncompliant {{Fix the syntax of this einops operation: the parameter b1 does not appear in the pattern.}}
