

def everything(bar, *args, somevar=True, **kwargs):
    print(bar, args, somevar, kwargs)

def empty_star(bar, *, somevar=True, **kwargs):
    print(bar, somevar, kwargs)

def after_star(bar, *args, boo, somevar=True, **kwargs):
    print(bar, boo, somevar, kwargs)

def start_empty_star(*, somevar=True):
    print(somevar)

if __name__ == '__main__':
    everything(1)
    empty_star(1)
    after_star(1, boo=2)
    start_empty_star(somevar=1)
