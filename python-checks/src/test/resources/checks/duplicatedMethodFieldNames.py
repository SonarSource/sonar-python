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

    NAME = foo(NAME)

    def NAME(self):
        pass
