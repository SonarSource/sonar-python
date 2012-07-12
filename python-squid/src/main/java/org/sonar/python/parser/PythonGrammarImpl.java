/*
 * Sonar Python Plugin
 * Copyright (C) 2011 SonarSource and Waleri Enns
 * dev@sonar.codehaus.org
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */
package org.sonar.python.parser;

import com.sonar.sslr.impl.matcher.GrammarFunctions;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.PythonTokenType;

import static com.sonar.sslr.api.GenericTokenType.EOF;
import static com.sonar.sslr.api.GenericTokenType.IDENTIFIER;
import static com.sonar.sslr.impl.matcher.GrammarFunctions.Predicate.not;
import static com.sonar.sslr.impl.matcher.GrammarFunctions.Standard.and;
import static com.sonar.sslr.impl.matcher.GrammarFunctions.Standard.o2n;
import static com.sonar.sslr.impl.matcher.GrammarFunctions.Standard.one2n;
import static com.sonar.sslr.impl.matcher.GrammarFunctions.Standard.opt;
import static com.sonar.sslr.impl.matcher.GrammarFunctions.Standard.or;
import static org.sonar.python.api.PythonTokenType.DEDENT;
import static org.sonar.python.api.PythonTokenType.INDENT;
import static org.sonar.python.api.PythonTokenType.NEWLINE;

/**
 * Based on http://docs.python.org/reference/grammar.html
 */
public class PythonGrammarImpl extends PythonGrammar {

  public PythonGrammarImpl() {
    expressions();
    simpleStatements();
    compoundStatements();
    toplevel();

    grammar();

    GrammarFunctions.enableMemoizationOfMatchesForAllRules(this);
  }

  private void grammar() {
    expression_stmt.is(
        testlist_star_expr,
        or(
            and(augassign, or(yield_expr, testlist)),
            and(o2n("=", or(yield_expr, testlist_star_expr)))));
    testlist_star_expr.is(or(test, star_expr), o2n(",", or(test, star_expr)), opt(","));
    augassign.is(or("+=", "-=", "*=", "/=", "//=", "%=", "**=", ">>=", "<<=", "&=", "^=", "|="));

    test.is(or(
        and(or_test, opt("if", or_test, "else", test)),
        lambdef));
    test_nocond.is(or(or_test, lambdef_nocond));
    lambdef.is("lambda", opt(varargslist), ":", test);
    lambdef_nocond.is("lambda", opt(varargslist), ":", test_nocond);

    star_expr.is("*", expr);
    expr.is(xor_expr, o2n("|", xor_expr));

    factor.is(or(
        and(or("+", "-", "~"), factor),
        power)).skipIfOneChild();
    power.is(atom, o2n(trailer), opt("**", factor));
    atom.is(or(
        and("(", opt(or(yield_expr, testlist_comp)), ")"),
        and("[", opt(testlist_comp), "]"),
        and("{", opt(dictorsetmaker), "}"),
        and("`", test, o2n(",", test), "`"),
        name,
        PythonTokenType.NUMBER,
        one2n(PythonTokenType.STRING),
        "...",
        PythonKeyword.NONE,
        PythonKeyword.TRUE,
        PythonKeyword.FALSE));
    testlist_comp.is(or(test, star_expr), or(comp_for, and(o2n(",", or(test, star_expr)), opt(","))));
    trailer.is(or(
        and("(", opt(arglist), ")"),
        and("[", subscriptlist, "]"),
        and(".", name)));
    subscriptlist.is(subscript, o2n(",", subscript), opt(","));
    subscript.is(or(
        and(".", ".", "."),
        and(opt(test), ":", opt(test), opt(sliceop)),
        test));
    sliceop.is(":", opt(test));
    exprlist.is(or(expr, star_expr), o2n(",", or(expr, star_expr)), opt(","));
    testlist.is(test, o2n(",", test), opt(","));
    dictorsetmaker.is(or(
        and(test, ":", test, or(comp_for, and(o2n(",", test, ":", test), opt(",")))),
        and(test, or(comp_for, and(o2n(",", test), opt(","))))));

    arglist.is(or(
            and(o2n(argument, ","), "*", test, o2n(",", argument), opt(",", "**", test)),
            and(o2n(argument, ","), "**", test),
            and(opt(argument, o2n(",", argument), opt(",")))));
    argument.is(or(
        and(test, "=", test),
        and(test, opt(comp_for))));
    comp_iter.is(or(comp_for, comp_if));
    comp_for.is("for", exprlist, "in", or_test, opt(comp_iter));
    comp_if.is("if", test_nocond, opt(comp_iter));

    yield_expr.is("yield", opt(testlist));

    name.is(IDENTIFIER);
    varargslist.is(or(
        and(o2n(fpdef, opt("=", test), ","), or(and("*", name, opt(",", "**", name)), and("**", name))),
        and(fpdef, opt("=", test), o2n(",", fpdef, opt("=", test)), opt(","))));
    fpdef.is(or(
        name,
        and("(", fplist, ")")));
    fplist.is(fpdef, o2n(",", fpdef), opt(","));
  }

  /**
   * Expressions
   * http://docs.python.org/reference/expressions.html
   */
  private void expressions() {
    m_expr.is(factor, o2n(or("*", "//", "/", "%"), factor)).skipIfOneChild();
    a_expr.is(m_expr, o2n(or("+", "-"), m_expr)).skipIfOneChild();

    shift_expr.is(a_expr, o2n(or("<<", ">>"), a_expr)).skipIfOneChild();

    and_expr.is(shift_expr, o2n("&", shift_expr)).skipIfOneChild();
    xor_expr.is(and_expr, o2n("^", and_expr)).skipIfOneChild();
    or_expr.is(xor_expr, o2n("|", xor_expr)).skipIfOneChild();

    comparison.is(or_expr, o2n(comp_operator, or_expr)).skipIfOneChild();
    comp_operator.is(or(
        "<",
        ">",
        "==",
        ">=",
        "<=",
        "!=",
        permissive_2_7("<>"),
        and("is", opt("not")),
        and(opt("not"), "in")));

    or_test.is(and_test, o2n("or", and_test)).skipIfOneChild();
    and_test.is(not_test, o2n("and", not_test)).skipIfOneChild();
    not_test.is(or(comparison, and("not", not_test))).skipIfOneChild();
  }

  /**
   * Simple statements
   * http://docs.python.org/reference/simple_stmts.html
   */
  private void simpleStatements() {
    simple_stmt.is(or(
        permissive_2_7(print_stmt),
        permissive_2_7(exec_stmt),
        expression_stmt,
        assert_stmt,
        pass_stmt,
        del_stmt,
        return_stmt,
        yield_stmt,
        raise_stmt,
        break_stmt,
        continue_stmt,
        import_stmt,
        global_stmt,
        nonlocal_stmt));

    print_stmt.is("print", not("("), or(
        and(">>", test, opt(one2n(",", test), opt(","))),
        and(opt(test, o2n(",", test), opt(",")))));

    exec_stmt.is("exec", not("("), expr, opt("in", test, opt(",", test)));

    assert_stmt.is("assert", test, opt(",", test));

    pass_stmt.is("pass");
    del_stmt.is("del", exprlist);
    return_stmt.is("return", opt(testlist));
    yield_stmt.is(yield_expr);
    raise_stmt.is("raise", opt(test, opt(or(and("from", test), permissive_2_7(",", test, opt(",", test))))));
    break_stmt.is("break");
    continue_stmt.is("continue");

    import_stmt.is(or(import_name, import_from));
    import_name.is("import", dotted_as_names);
    import_from.is("from", or(and(o2n("."), dotted_name), one2n(".")), "import", or("*", and("(", import_as_names, ")"), import_as_names));
    import_as_name.is(name, opt("as", name));
    dotted_as_name.is(dotted_name, opt("as", name));
    import_as_names.is(import_as_name, o2n(",", import_as_name), opt(","));
    dotted_as_names.is(dotted_as_name, o2n(",", dotted_as_name));

    global_stmt.is("global", name, o2n(",", name));
    nonlocal_stmt.is("nonlocal", name, o2n(",", name));
  }

  /**
   * Compound statements
   * http://docs.python.org/reference/compound_stmts.html
   */
  private void compoundStatements() {
    compound_stmt.is(or(
        if_stmt,
        while_stmt,
        for_stmt,
        try_stmt,
        with_stmt,
        funcdef,
        classdef));
    suite.is(or(
        and(stmt_list, NEWLINE),
        and(NEWLINE, INDENT, one2n(statement), DEDENT)));
    statement.is(or(
        and(stmt_list, NEWLINE),
        compound_stmt,
        permissive(stmt_list)));
    stmt_list.is(simple_stmt, o2n(";", simple_stmt), opt(";"));

    if_stmt.is("if", test, ":", suite, o2n("elif", test, ":", suite), opt("else", ":", suite));
    while_stmt.is("while", test, ":", suite, opt("else", ":", suite));
    for_stmt.is("for", exprlist, "in", testlist, ":", suite, opt("else", ":", suite));

    try_stmt.is("try", ":", suite, or(and(o2n(except_clause, ":", suite), opt("else", ":", suite), opt("finally", ":", suite)), and("finally", ":", suite)));
    except_clause.is("except", opt(test, opt(or("as", ","), test)));

    with_stmt.is("with", with_item, o2n(",", with_item), ":", suite);
    with_item.is(test, opt("as", expr));

    funcdef.is(opt(decorators), "def", funcname, "(", opt(varargslist), ")", ":", suite);
    funcname.is(name);

    decorators.is(one2n(decorator));
    decorator.is("@", dotted_name, opt("(", opt(arglist), ")"), NEWLINE);
    dotted_name.is(name, o2n(".", name));

    classdef.is(opt(decorators), "class", classname, opt("(", opt(arglist), ")"), ":", suite);
    classname.is(name);
  }

  /**
   * Top-level components
   * http://docs.python.org/reference/toplevel_components.html
   */
  private void toplevel() {
    file_input.is(o2n(or(NEWLINE, statement)), EOF);
  }

  /**
   * Syntactic sugar, which helps to specify constructs, which are not part of http://docs.python.org/reference/grammar
   */
  private static Object permissive(Object object) {
    return object;
  }

  /**
   * Syntactic sugar, which helps to specify constructs, which are part of Python 2.7.3, but not 3.0.
   */
  private static Object permissive_2_7(Object... object) {
    return and(object);
  }

}
