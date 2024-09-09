from torch.utils.data import DataLoader
from torch.utils.data import DataLoader as AliasedDataLoader
import torch.utils.data
import os

train_dataset = ...

noncomp = DataLoader(dataset=train_dataset, batch_size=32) # Noncompliant {{Specify the `num_workers` parameter.}}
         #^^^^^^^^^^

noncomp = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32) # Noncompliant
         #^^^^^^^^^^^^^^^^^^^^^^^^^^^

noncomp = AliasedDataLoader(dataset=train_dataset, batch_size=32) # Noncompliant
         #^^^^^^^^^^^^^^^^^

noncomp = DataLoader() # Noncompliant

comp1 = DataLoader(dataset=train_dataset, batch_size=32, num_workers=len(train_dataset) / os.cpu_count())
comp2 = DataLoader(dataset=train_dataset, batch_size=32, num_workers=0)
comp3 = DataLoader(dataset=train_dataset, batch_size=32, num_workers=1)
comp4 = DataLoader(train_dataset, 32, False, False, False, 3) # the num_workers is the 6th arg, and in this case `3`
comp5 = DataLoader(train_dataset, 32, False, False, False, 3, False)

dict = {"someStuff":4}
comp5 = DataLoader(**dict)
comp6 = DataLoader(dataset=train_dataset, **dict)
comp7 = DataLoader(**{"someStuff": 3})

list = [1, 2, 3, 4, 5, 6]
comp8 = DataLoader(*list)
comp8 = DataLoader(dataset=train_dataset, *list)
comp9 = DataLoader(*[1, 2, 3])

comp10 = DataLoader(dataset=train_dataset, num_workers=None)

class SubDataLoader(DataLoader):
    pass

# this should raise an issue but this is currently not supported
comp10 = SubDataLoader()