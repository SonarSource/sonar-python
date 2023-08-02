class C:
    def returnsSelf01(self):
        return self

    def returnsSelf02(this, self):
        return self

    # The implementation of isSelf is rather simple and does not check the method context.
    # I.e. in this case it does not check that the surrounding method is static and what is being returned is not actually "self" in the
    #   usual sense.
    @staticmethod
    def returnsSelf03(self):
        return self

    # Similar issue as above:
    @classmethod
    def returnsSelf04(self):
        return self

    def returnsSthElse01(this, self):
        return this

    def returnsSthElse02(self):
        return "Hello World"

# This is another case where the implementation does not check the context, i.e. that the surrounding function is not actually a method.
# See also returnsSelf03.
def returnsSelf05(self):
    return self
