/*
 * SonarQube Python Plugin
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
package org.sonar.python.api;

import com.sonar.sslr.api.Grammar;
import com.sonar.sslr.api.Rule;

public class PythonGrammar extends Grammar {

  public Rule factor;
  public Rule trailer;
  public Rule subscriptlist;
  public Rule subscript;
  public Rule sliceop;
  public Rule testlist_comp;
  public Rule dictorsetmaker;

  public Rule arglist;
  public Rule argument;

  public Rule name;
  public Rule varargslist;
  public Rule fpdef;
  public Rule fplist;

  public Rule test;
  public Rule testlist;

  public Rule comp_for;
  public Rule comp_iter;
  public Rule comp_if;
  public Rule test_nocond;
  public Rule exprlist;
  public Rule expr;
  public Rule star_expr;

  public Rule testlist_star_expr;

  public Rule yield_expr;

  // Expressions

  public Rule atom;

  public Rule power;

  public Rule a_expr;
  public Rule m_expr;

  public Rule shift_expr;

  public Rule xor_expr;
  public Rule and_expr;
  public Rule or_expr;

  public Rule comparison;
  public Rule comp_operator;

  public Rule or_test;
  public Rule and_test;
  public Rule not_test;

  public Rule lambdef;
  public Rule lambdef_nocond;

  // Simple statements

  public Rule simple_stmt;
  public Rule expression_stmt;
  public Rule print_stmt;
  public Rule exec_stmt;
  public Rule assert_stmt;

  public Rule augassign;

  public Rule pass_stmt;
  public Rule del_stmt;
  public Rule return_stmt;
  public Rule yield_stmt;
  public Rule raise_stmt;
  public Rule break_stmt;
  public Rule continue_stmt;

  public Rule import_stmt;
  public Rule import_name;
  public Rule import_from;
  public Rule import_as_name;
  public Rule dotted_as_name;
  public Rule import_as_names;
  public Rule dotted_as_names;

  public Rule global_stmt;
  public Rule nonlocal_stmt;

  // Compound statements

  public Rule compound_stmt;
  public Rule suite;
  public Rule statement;
  public Rule stmt_list;

  public Rule if_stmt;
  public Rule while_stmt;
  public Rule for_stmt;

  public Rule try_stmt;
  public Rule except_clause;

  public Rule with_stmt;
  public Rule with_item;

  public Rule funcdef;
  public Rule decorators;
  public Rule decorator;
  public Rule dotted_name;
  public Rule funcname;

  public Rule classdef;
  public Rule classname;

  // Top-level components

  public Rule file_input;

  @Override
  public Rule getRootRule() {
    return file_input;
  }

}
