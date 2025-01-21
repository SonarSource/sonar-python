class Rectangle(object):

    @classmethod
    def area(bob, height, width):  #Noncompliant {{Add 'cls' as first argument.}}
        return height * width
