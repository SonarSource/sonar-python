class A:
    def go_up(self):
        pass

    def GO_UP(self):   # Noncompliant {{Rename method "GO_UP" to prevent any misunderstanding/clash with method "go_up" defined on line 2}}
#       ^^^^^
        self.go_down = 2

    def GO_DOWN(self):  # Noncompliant [[secondary=-2]]
        pass

    def NAME(self):
        pass

    NAME = func(NAME)

    def NAME(self):
        pass

    def hello(self):
        pass

    foo = 1
    
    fOo = 1            # Noncompliant {{Rename field "fOo" to prevent any misunderstanding/clash with field "foo" defined on line 23}}
#   ^^^

    foO: int = 1       # Noncompliant {{Rename field "foO" to prevent any misunderstanding/clash with field "fOo" defined on line 25}}
#   ^^^

    baz = 100

    bat = 1000

    class B:

        Foo = 1

        bar = 10

        def __init__(self):
            self.baR = 10      # Noncompliant {{Rename field "baR" to prevent any misunderstanding/clash with field "bar" defined on line 39}}
#                ^^^
            self.FoO: int = 10 # Noncompliant
#                ^^^
            self.baz = 100
            self.baT = 1000

        def HEllo(self):
            pass
