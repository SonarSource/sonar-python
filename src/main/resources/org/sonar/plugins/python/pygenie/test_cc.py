from cc import measure_complexity


def dedent(s):
    """Dedent a suite of code so that the parser does not choke."""
    retval = []
    sp = None
    for line in (s.rstrip() for s in s.split('\n')):
        if not line:
            retval.append('')
        else:
            if sp is None:
                sp = len(line) - len(line.lstrip())
            if line[:sp] != sp * ' ':
                raise IndentationError('bad dedent')
            retval.append(line[sp:])
    return '\n'.join(retval)


simple_complexity_snippets = [
        (3, dedent("""
                if a and b:
                    pass
                else:
                    pass
            """)
        ),
        (5, dedent("""
                if a and b:
                    pass
                elif c and d:
                    pass
                else:
                    pass
            """)
        ),
        (5, dedent("""
                if (a and b) or (c and d):
                    pass
                else:
                    pass
            """)
        ),
        (4, dedent("""
                if a and b or c:
                    pass
                else:
                    pass
            """)
        ),
        (2, dedent("""
                for x in xrange(10):
                    print x
                else:
                    print 'blah'
            """)
        ),
        (2, dedent("""
                while a < 100:
                    print a
                else:
                    print 'blah'
            """)
        ),
        (3, dedent("""
                while a > 10 and b < 10:
                    print 'whoa'
            """)
        ),
        (3, dedent("""
                while a > 10 and b < 10:
                    print 'whoa'
                else:
                    print 'yeppers'
            """)
        ),
        (5, dedent("""
                while (a and b) or (c and d):
                    pass
                else:
                    pass
            """)
        ),
        (4, dedent("""
                while a and b or c:
                    pass
                else:
                    pass
            """)
        ),
        (2, dedent("""
                [x for x in some_list]
            """)
        ),
        (3, dedent("""
                [x for x in some_list if x.strip()]
            """)
        ),
        (2, dedent("""
                (x for x in some_list)
            """)
        ),
        (3, dedent("""
                (x for x in some_list if x.strip())
            """)
        ),
]


def test_snippets():
    def _assert(expected_complexity, code):
        stats = measure_complexity(code)
        print stats.complexity, expected_complexity
        assert stats.complexity == expected_complexity
    for snippet in simple_complexity_snippets:
        yield _assert, snippet[0], snippet[1]


def test_empty_def():
    code = dedent("""
        def f():
            pass
    """)
    stats = measure_complexity(code)
    assert stats.complexity == 1
    assert stats.functions[0].complexity == 1


def test_def():
    code = dedent('''
            def f(a, b, c):
                if a:
                    pass
                else:
                    pass
    ''')
    stats = measure_complexity(code)
    assert stats.classes == []
    assert len(stats.functions) == 1
    assert stats.functions[0].name == 'f'
    assert stats.functions[0].complexity == 2


def test_module():
    code = dedent('''
            if True:
                pass
            else:
                pass
    ''')
    stats = measure_complexity(code)
    assert stats.name == '<module>'
    assert stats.classes == []
    assert stats.functions == []
    assert stats.complexity == 2


def test_class():
    code = dedent('''
            class A:
                if True:
                    pass
                def f(self):
                    if False:
                        pass
    ''')
    stats = measure_complexity(code)
    assert stats.name == '<module>'
    assert stats.classes[0].name == 'A'
    assert stats.classes[0].complexity == 2


def test_loops():
    code = dedent('''
        def f(a):
            while a < 10:
                print a

        def g(a):
            for x in range(a):
                print a

        def h(a):
            while a < 10:
                for x in range(a):
                    if x % 2:
                        print 'odd'
    ''')
    stats = measure_complexity(code)
    f, g, h = stats.functions
    assert f.complexity == 2
    assert g.complexity == 2
    assert h.complexity == 4


def test_lambdas():
    code = dedent('''
        incr = lambda x: x + 1
        decr = lambda x: x - 1

        def f():
            incr = lambda x: x + 1
            decr = lambda x: x - 1
    ''')
    stats = measure_complexity(code)
    incr, decr, f = stats.functions
    assert incr.complexity == 1
    assert decr.complexity == 1


def test_nested_classes():
    code = dedent('''
        class A:
            if True:
                pass

            class Inner:
                if False:
                    pass

    ''')
    stats = measure_complexity(code)
    assert stats.classes[0].name == 'A'
    assert stats.classes[0].complexity == 2
    assert stats.classes[0].classes[0].name == 'Inner'
    assert stats.classes[0].classes[0].complexity == 2

