def myfunc():
    def unused() -> str:
        smth = 42
        max = 24
        if smth == smth:
            print("Hello?")
        return smth

    print("Hello!")


def my_other_func():
    my_list = []
    if my_list is []:
        print("Impossible!")
    my_tuple = (1, 2)
    try:
        my_tuple + my_list
    except TypeError:
        print("Hello there!")
    return 24


def main():
    myfunc()
    my_other_func()


if __name__ == "__main__":
    main()
