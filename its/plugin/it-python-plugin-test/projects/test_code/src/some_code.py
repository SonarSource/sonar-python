def foo():
    assert (a, b)  # Issue here (S5905 runs on all files)
    if True:  # Issue here (S3923 runs on main files)
        print("hello")
    else:
        print("hello")
