/*
 * Sonar Python Plugin
 * Copyright (C) 2011 Waleri Enns
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

import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonTokenType;

import static com.sonar.sslr.api.GenericTokenType.EOF;
import static com.sonar.sslr.api.GenericTokenType.IDENTIFIER;
import static com.sonar.sslr.impl.matcher.GrammarFunctions.Standard.and;
import static com.sonar.sslr.impl.matcher.GrammarFunctions.Standard.o2n;
import static com.sonar.sslr.impl.matcher.GrammarFunctions.Standard.one2n;
import static com.sonar.sslr.impl.matcher.GrammarFunctions.Standard.opt;
import static com.sonar.sslr.impl.matcher.GrammarFunctions.Standard.or;
import static org.sonar.python.api.PythonTokenType.DEDENT;
import static org.sonar.python.api.PythonTokenType.INDENT;
import static org.sonar.python.api.PythonTokenType.NEWLINE;

/**
 * Based on http://docs.python.org/release/3.2/reference/index.html
 */
public class PythonGrammarImpl extends PythonGrammar {

  public PythonGrammarImpl() {
    expressions();
    simpleStatements();
    compoundStatements();
    toplevel();
  }

  /**
   * Expressions
   * http://docs.python.org/release/3.2/reference/expressions.html
   */
  private void expressions() {
    comprehension.mock();

    // attributeref.is(primary, ".", IDENTIFIER);
    attributeref.mock();

    // subscription.is(primary, "[", expression_list, "]");
    subscription.mock();

    // slicing ::= primary "[" slice_list "]"
    // slice_list ::= slice_item ("," slice_item)* [","]
    // slice_item ::= expression | proper_slice
    // proper_slice ::= [lower_bound] ":" [upper_bound] [ ":" [stride] ]
    // lower_bound ::= expression
    // upper_bound ::= expression
    // stride ::= expression
    slicing.mock();

    // call.is(primary, "(", opt(or(and(argument_list, opt(",")), comprehension)), ")");
    call.mock();
    argument_list.is(or(
        and(positional_arguments, opt(",", keyword_arguments), opt(",", "*", expression), opt(",", keyword_arguments), opt(",", "**", expression)),
        and(keyword_arguments, opt(",", "*", expression), opt(",", keyword_arguments), opt(",", "**", expression)),
        and("*", expression, opt(",", keyword_arguments), opt(",", "**", expression)),
        and("**", expression)));
    positional_arguments.is(expression, o2n(",", expression));
    keyword_arguments.is(keyword_item, o2n(",", keyword_item));
    keyword_item.is(IDENTIFIER, "=", expression);

    // literal ::= stringliteral | bytesliteral | integer | floatnumber | imagnumber
    literal.is(or(PythonTokenType.STRING, PythonTokenType.NUMBER));

    atom.is(or(
        IDENTIFIER,
        literal,
        enclosure));

    // enclosure ::= parenth_form | list_display | dict_display | set_display | generator_expression | yield_atom
    enclosure.mock();

    // primary.is(or(
    // atom,
    // attributeref,
    // subscription,
    // slicing,
    // call));

    // power.is(primary, opt("**", u_expr));

    // Next 10 rules were taken from http://docs.python.org/reference/grammar.html
    power.is(atom, o2n(trailer), opt("**", factor));
    factor.is(or(
        and(or("+", "-", "~"), factor),
        power));
    trailer.is(or(
        and("(", opt(argument_list), ")"),
        and("[", subscriptlist, "]"),
        and(".", IDENTIFIER)));
    subscriptlist.is(subscript, o2n(",", subscript), opt(","));
    subscript.is(or(
        and(".", ".", "."),
        conditional_expression,
        and(opt(conditional_expression), ":", opt(conditional_expression), opt(sliceop))));
    sliceop.is(":", opt(conditional_expression));
    atom.override(or(
        and("(", opt(or(yield_expression, testlist_comp)), ")"),
        and("[", opt(listmaker), "]"),
        and("{", opt(dictorsetmaker), "}"),
        // '`' testlist1 '`' |
        IDENTIFIER,
        literal));
    listmaker.is(
        conditional_expression,
        or( /* list_for , */and(o2n(",", conditional_expression), opt(","))));
    testlist_comp.is(conditional_expression, or(/* comp_for, */and(o2n(",", conditional_expression), opt(","))));
    dictorsetmaker.is(or(
        and(conditional_expression, ":", conditional_expression, or(/* comp_for, */ and(o2n(conditional_expression, ":", conditional_expression), opt(",")))),
        and(conditional_expression, or(/* comp_for, */and(o2n(",", conditional_expression), opt(","))))));

    u_expr.is(or(
        and("-", u_expr),
        and("+", u_expr),
        and("~", u_expr),
        power)).skipIfOneChild();

    m_expr.is(u_expr, o2n(or("*", "//", "/", "%"), u_expr)).skipIfOneChild();
    a_expr.is(m_expr, o2n(or("+", "-"), m_expr)).skipIfOneChild();

    shift_expr.is(a_expr, o2n(or("<<", ">>"), a_expr)).skipIfOneChild();

    and_expr.is(shift_expr, o2n("^", shift_expr)).skipIfOneChild();
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
        and("is", opt("not")),
        and(opt("not"), "in")));

    or_test.is(and_test, o2n("or", and_test)).skipIfOneChild();
    and_test.is(not_test, o2n("and", not_test)).skipIfOneChild();
    not_test.is(or(comparison, and("not", not_test))).skipIfOneChild();

    conditional_expression.is(or_test, opt("if", or_test, "else", expression)).skipIfOneChild();
    expression.is(or(conditional_expression, lambda_form));
    expression_nocond.is(or(or_test, lambda_form_nocond));

    lambda_form.is("lambda", opt(parameter_list), ":", expression);
    lambda_form_nocond.is("lambda", opt(parameter_list), ":", expression_nocond);

    expression_list.is(expression, o2n(",", expression), opt(","));

    yield_expression.is("yield", opt(expression_list));
  }

  /**
   * Simple statements
   * http://docs.python.org/release/3.2/reference/simple_stmts.html
   */
  private void simpleStatements() {
    simple_stmt.is(or(
        assert_stmt,
        assignment_stmt,
        pass_stmt,
        del_stmt,
        return_stmt,
        yield_stmt,
        raise_stmt,
        break_stmt,
        continue_stmt,
        import_stmt,
        global_stmt,
        nonlocal_stmt,
        expression_stmt,
        augmented_assignment_stmt));

    expression_stmt.is(expression_list);
    assert_stmt.is("assert", expression, opt(",", expression));

    assignment_stmt.is(one2n(target_list, "="), or(expression_list, yield_expression));
    target_list.is(target, o2n(",", target), opt(","));
    target.is(or(
        IDENTIFIER,
        and("(", target_list, ")"),
        and("[", target_list, "]"),
        attributeref,
        subscription,
        slicing,
        and("*", target)));

    augmented_assignment_stmt.is(augtarget, augop, or(expression_list, yield_expression));
    augtarget.is(or(IDENTIFIER, attributeref, subscription, slicing));
    augop.is(or("+=", "-=", "*=", "/=", "//=", "%=", "**=", ">>=", "<<=", "&=", "^=", "|="));

    pass_stmt.is("pass");
    del_stmt.is("del", target_list);
    return_stmt.is("return", opt(expression_list));
    yield_stmt.is(yield_expression);
    raise_stmt.is("raise", opt(expression, opt("from", expression)));
    break_stmt.is("break");
    continue_stmt.is("continue");

    import_stmt.is(or(
        and("import", module, opt("as", name), o2n(",", module, opt("as", name))),
        and("from", relative_module, "import", IDENTIFIER, opt("as", name), o2n(",", IDENTIFIER, opt("as", name))),
        and("from", relative_module, "import", "(", IDENTIFIER, opt("as", name), o2n(",", IDENTIFIER, opt("as", name)), opt(","), ")"),
        and("from", module, "import", "*")));
    module.is(o2n(IDENTIFIER, "."), IDENTIFIER);
    relative_module.is(or(
        and(o2n("."), module),
        one2n(".")));
    name.is(IDENTIFIER);

    global_stmt.is("global", IDENTIFIER, o2n(",", IDENTIFIER));
    nonlocal_stmt.is("nonlocal", IDENTIFIER, o2n(",", IDENTIFIER));
  }

  /**
   * Compound statements
   * http://docs.python.org/release/3.2/reference/compound_stmts.html
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
        compound_stmt));
    stmt_list.is(simple_stmt, o2n(";", simple_stmt), opt(";"));

    if_stmt.is("if", expression, ":", suite, o2n("elif", expression, ":", suite), opt("else", ":", suite));
    while_stmt.is("while", expression, ":", suite, opt("else", ":", suite));
    for_stmt.is("for", target_list, "in", expression_list, ":", suite, opt("else", ":", suite));

    try_stmt.is(or(try1_stmt, try2_stmt));
    try1_stmt.is("try", ":", suite,
        one2n("except", opt(expression, opt("as", target)), ":", suite),
        opt("else", ":", suite),
        opt("finally", ":", suite));
    try2_stmt.is("try", ":", suite, "finally", ":", suite);

    with_stmt.is("with", with_item, o2n(",", with_item), ":", suite);
    with_item.is(expression, opt("as", target));

    funcdef.is(opt(decorators), "def", funcname, "(", opt(parameter_list), ")", opt("->", expression), ":", suite);
    decorators.is(one2n(decorator));
    decorator.is("@", dotted_name, opt("(", opt(argument_list, opt(",")), ")"), NEWLINE);
    dotted_name.is(IDENTIFIER, o2n(".", IDENTIFIER));
    parameter_list.is(
        o2n(defparameter, ","),
        or(
            and("*", opt(parameter), o2n(",", defparameter), opt(",", "**", parameter)),
            and("**", parameter),
            and(defparameter, opt(","))));
    parameter.is(IDENTIFIER, opt(":", expression));
    defparameter.is(parameter, opt("=", expression));
    funcname.is(IDENTIFIER);

    classdef.is(opt(decorators), "class", classname, opt(inheritance), ":", suite);
    inheritance.is("(", opt(or(and(argument_list, opt(",")), comprehension)), ")");
    classname.is(IDENTIFIER);
  }

  /**
   * Top-level components
   * http://docs.python.org/release/3.2/reference/toplevel_components.html
   */
  private void toplevel() {
    file_input.is(o2n(or(NEWLINE, statement)), EOF);
  }

}
