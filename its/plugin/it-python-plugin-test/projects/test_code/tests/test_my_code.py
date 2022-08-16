def test_something():
    assert 1 == 2
    assert (a, b)  # Issue here (S5905)
    if True:  # No issue here (S3923 doesn't run on test files)
      print("hello")
    else:
      print("hello")
