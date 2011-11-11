#!/usr/bin/env python

import compiler
from compiler.visitor import ASTVisitor


class Stats(object):

    def __init__(self, name):
        self.name = name
        self.classes = []
        self.functions = []
        self.complexity = 1

    def __str__(self):
        return 'Stats: name=%r, classes=%r, functions=%r, complexity=%r' \
                % (self.name, self.classes, self.functions, self.complexity)

    __repr__ = __str__


class ClassStats(Stats):

    def __str__(self):
        return 'Stats: name=%r, methods=%r, complexity=%r, inner_class=%r' \
                % (self.name, self.functions, self.complexity, self.classes)

    __repr__ = __str__


class DefStats(Stats):

    def __str__(self):
        return 'DefStats: name=%r, complexity=%r' \
                % (self.name, self.complexity)

    __repr__ = __str__


class CCVisitor(ASTVisitor):
    """Encapsulates the cyclomatic complexity counting."""

    def __init__(self, ast, stats=None, description=None):
        ASTVisitor.__init__(self)
        if isinstance(ast, basestring):
            ast = compiler.parse(ast)

        self.stats = stats or Stats(description or '<module>')
        for child in ast.getChildNodes():
            compiler.walk(child, self, walker=self)

    def dispatchChildren(self, node):
        for child in node.getChildNodes():
            self.dispatch(child)

    def visitFunction(self, node):
        if not hasattr(node, 'name'): # lambdas
            node.name = '<lambda>'
        stats = DefStats(node.name)
        stats = CCVisitor(node, stats).stats
        self.stats.functions.append(stats)

    visitLambda = visitFunction

    def visitClass(self, node):
        stats = ClassStats(node.name)
        stats = CCVisitor(node, stats).stats
        self.stats.classes.append(stats)

    def visitIf(self, node):
        self.stats.complexity += len(node.tests)
        self.dispatchChildren(node)

    def __processDecisionPoint(self, node):
        self.stats.complexity += 1
        self.dispatchChildren(node)

    visitFor = visitGenExprFor = visitGenExprIf \
            = visitListCompFor = visitListCompIf \
            = visitWhile = _visitWith = __processDecisionPoint

    def visitAnd(self, node):
        self.dispatchChildren(node)
        self.stats.complexity += 1

    def visitOr(self, node):
        self.dispatchChildren(node)
        self.stats.complexity += 1


def measure_complexity(ast, module_name=None):
    return CCVisitor(ast, description=module_name).stats


class Table(object):

    def __init__(self, headings, rows):
        self.headings = headings
        self.rows = rows

        max_col_sizes = [len(x) for x in headings]
        for row in rows:
            for i, col in enumerate(row):
                max_col_sizes[i] = max(max_col_sizes[i], len(str(col)))
        self.max_col_sizes = max_col_sizes

    def __iter__(self):
        for row in self.rows:
            yield row

    def __nonzero__(self):
        return len(self.rows)


class PrettyPrinter(object):

    def __init__(self, out, verbose=False):
        self.out = out
        self.verbose = verbose

    def pprint(self, filename, stats):
        self.out.write('File: %s\n' % filename)

        stats = self.flatten_stats(stats)

        if not self.verbose:
            # filter out suites with low complexity numbers
            stats = (row for row in stats if row[-1] > 7)

        stats = sorted(stats, lambda a, b: cmp(b[2], a[2]))

        table = Table(['Type', 'Name', 'Complexity'], stats)
        if table:
            self.pprint_table(table)
        else:
            self.out.write('This code looks all good!\n')
        self.out.write('\n')

    def pprint_table(self, table):
        for n, col in enumerate(table.headings):
            self.out.write(str(col).ljust(table.max_col_sizes[n] + 1))
        self.out.write('\n')
        self.out.write('-' * (sum(table.max_col_sizes) + len(table.headings) - 1) + '\n')
        for row in table:
            for n, col in enumerate(row):
                self.out.write(str(col).ljust(table.max_col_sizes[n] + 1))
            self.out.write('\n')

    def flatten_stats(self, stats):
        def flatten(stats, ns=None):
            if not ns:
                yield 'X', stats.name, stats.complexity
            for s in stats.classes:
                name = '.'.join(filter(None, [ns, s.name]))
                yield 'C', name, s.complexity
                for x in s.functions:
                    fname = '.'.join([name, x.name])
                    yield 'M', fname, x.complexity
            for s in stats.functions:
                name = '.'.join(filter(None, [ns, s.name]))
                yield 'F', name, s.complexity
        
        return [t for t in flatten(stats)]


