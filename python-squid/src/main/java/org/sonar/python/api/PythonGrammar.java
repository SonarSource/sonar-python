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
package org.sonar.python.api;

import com.sonar.sslr.api.Grammar;
import com.sonar.sslr.api.Rule;

public class PythonGrammar extends Grammar {

  public Rule yield_expression;

  // TODO
  public Rule factor;
  public Rule trailer;
  public Rule subscriptlist;
  public Rule subscript;
  public Rule sliceop;
  public Rule listmaker;
  public Rule testlist_comp;

  // Expressions

  public Rule literal;
  public Rule enclosure;

  public Rule atom;
  public Rule attributeref;
  public Rule subscription;
  public Rule slicing;

  public Rule call;
  public Rule argument_list;
  public Rule comprehension;
  public Rule keyword_arguments;
  public Rule positional_arguments;
  public Rule keyword_item;

  // public Rule primary;

  public Rule power;

  public Rule u_expr;

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

  public Rule expression;
  public Rule expression_nocond;
  public Rule conditional_expression;

  public Rule lambda_form;
  public Rule lambda_form_nocond;

  public Rule expression_list;

  // Simple statements

  public Rule simple_stmt;
  public Rule expression_stmt;
  public Rule assert_stmt;

  public Rule assignment_stmt;
  public Rule target_list;
  public Rule target;

  public Rule augmented_assignment_stmt;
  public Rule augtarget;
  public Rule augop;

  public Rule pass_stmt;
  public Rule del_stmt;
  public Rule return_stmt;
  public Rule yield_stmt;
  public Rule raise_stmt;
  public Rule break_stmt;
  public Rule continue_stmt;

  public Rule import_stmt;
  public Rule module;
  public Rule relative_module;
  public Rule name;

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
  public Rule try1_stmt;
  public Rule try2_stmt;

  public Rule with_stmt;
  public Rule with_item;

  public Rule funcdef;
  public Rule decorators;
  public Rule decorator;
  public Rule dotted_name;
  public Rule parameter_list;
  public Rule parameter;
  public Rule defparameter;
  public Rule funcname;

  public Rule classdef;
  public Rule inheritance;
  public Rule classname;

  // Top-level components

  public Rule file_input;

  @Override
  public Rule getRootRule() {
    return file_input;
  }

}
