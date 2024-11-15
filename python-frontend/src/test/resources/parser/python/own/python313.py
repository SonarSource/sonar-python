# Tests Python 3.13 features

# annotation scope
class C[T]:
    type Alias1 = lambda: T
    type Alias2 = [i for i in range(3)]
    type Alias3 = {i: True for i in range(3)}
    type Alias4 = {i or i in range(3)}

class name_2[*name_5, name_3: int]:
    (name_3 := name_4)

    class name_4[name_5: name_5]((name_4 for name_5 in name_0 if name_3), name_2 if name_3 else name_0):
        pass


# global in except block
a=5

def f():
    try:
        pass
    except:
        global a
    else:
        print(a)
