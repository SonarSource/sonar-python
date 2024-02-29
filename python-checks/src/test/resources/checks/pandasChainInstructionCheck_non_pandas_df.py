class DataFrame:
    def foo(self):
        return self

def non_pandas_data_frame_chain():
    df = DataFrame()
    df2 = df.foo().foo().foo().foo().foo().foo().foo().foo() # OK
