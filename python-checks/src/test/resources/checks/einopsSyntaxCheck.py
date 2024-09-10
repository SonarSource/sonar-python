import torch
import einops
from einops import rearrange, reduce, repeat

img = torch.randn(32, 32, 3)

rearrange(img, 'h w c -> c h w')

imgs = torch.randn(10, 32, 32, 3)

rearrange(imgs, 'b h w c -> b c h w')
rearrange(imgs, 'b h w c -> (b h) w c')

unstacked = rearrange(imgs, '(b h) w c -> b h w c', b=10)

unstacked2 = rearrange(imgs, '(... h) w c -> ... h w c') # Noncompliant {{Fix the syntax of this einops.rearrange operation: Ellipsis inside parenthesis in the left side is not allowed}}
                            #^^^^^^^^^^^^^^^^^^^^^^^^^^
unstacked2 = einops.rearrange(imgs, '(... h) w c -> ... h w c') # Noncompliant 
                                   #^^^^^^^^^^^^^^^^^^^^^^^^^^
unstacked2 = einops.rearrange(pattern='(... h) w c -> ... h w c', tensor=imgs) # Noncompliant 
                                     #^^^^^^^^^^^^^^^^^^^^^^^^^^
unstacked2 = rearrange(imgs, '(... h) w c -> (... h) w c') # Noncompliant
                            #^^^^^^^^^^^^^^^^^^^^^^^^^^^^
unstacked2 = rearrange(imgs, '... h w c -> (... h) w c')
