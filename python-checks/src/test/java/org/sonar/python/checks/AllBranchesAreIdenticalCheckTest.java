/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks;

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class AllBranchesAreIdenticalCheckTest {

  private final PythonCheck check = new AllBranchesAreIdenticalCheck();
  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/allBranchesAreIdentical.py", check);
  }

  @Test
  void quickfix_one_statement() {
    String noncompliant =
      "def func():\n" +
      "    if b == 0:\n" +
      "        doSomething()\n" +
      "    else:\n" +
      "        doSomething()\n";
    String fixed =
      "def func():\n" +
      "    doSomething()\n";
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
  }

  @Test
  void quickfix_semicolons() {
    String noncompliant =
      "def func():\n" +
      "    if b == 0:\n" +
      "        doSomething(); doOneMoreThing()\n"+
      "    else:\n" +
      "        doSomething(); doOneMoreThing()\n";
    String fixed =
      "def func():\n" +
      "    doSomething(); doOneMoreThing()\n";
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
  }

  @Test
  void if_enclosed() {
    String noncompliant =
      "def func():\n" +
      "    if b == 0:\n" +
      "        if a == 1:\n"+
      "            doSomething()\n"+
      "    else:\n" +
      "        if a == 1:\n"+
      "            doSomething()\n";
    String fixed =
      "def func():\n" +
      "    if a == 1:\n"+
      "        doSomething()\n";
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
  }


  @Test
  void oneline() {
    String noncompliant = "a = 1 if x else 1";
    String fixed = "a = 1";
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
  }

  @Test
  void no_quick_fix_with_side_effect_in_first_condition() {
    PythonQuickFixVerifier.verifyNoQuickFixes(check,
      "if foo():\n" +
        "    doSomething()\n" +
        "else:\n" +
        "    doSomething()\n"
    );
  }

  @Test
  void no_quick_fix_with_side_effect_within_operator_right_hand() {
    PythonQuickFixVerifier.verifyNoQuickFixes(check,
      "if 1 == 2 and foo():\n" +
        "    doSomething()\n" +
        "else:\n" +
        "    doSomething()\n"
    );
  }

  @Test
  void no_quick_fix_with_side_effect_within_operator_left_hand() {
    PythonQuickFixVerifier.verifyNoQuickFixes(check,
      "if foo() and 1 == 2:\n" +
        "    doSomething()\n" +
        "else:\n" +
        "    doSomething()\n"
    );
  }

  @Test
  void no_quick_fix_with_side_effect_within_operator_parenthesis() {
    PythonQuickFixVerifier.verifyNoQuickFixes(check,
      "if 1 == 3 or (foo() and 1 == 2):\n" +
        "    doSomething()\n" +
        "else:\n" +
        "    doSomething()\n"
    );
  }

  @Test
  void no_quick_fix_with_side_effect_in_elif_condition() {
    PythonQuickFixVerifier.verifyNoQuickFixes(check,
      "if b == 0:\n" +
        "    doSomething()\n" +
        "elif bar():\n" +
        "    doSomething()\n" +
        "else:\n" +
        "    doSomething()\n"
    );
  }

  @Test
  void test_multiple_statement(){
    String noncompliant ="def func():\n" +
      "    if b == 0:\n" +
      "        doSomething()\n" +
      "        doOneMoreThing()\n" +
      "    else:\n" +
      "        doSomething()\n" +
      "        doOneMoreThing()\n";
    String fixed = "def func():\n" +
      "    doSomething()\n" +
      "    doOneMoreThing()\n";
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
  }

  @Test
  void comments() {
    String noncompliant = "if a == 0:\n" +
      "    # true branch comment1\n" +
      "    doSomething()  # true branch comment2\n" +
      "    # true branch comment3\n" +
      "else:\n" +
      "    # false branch comment1\n" +
      "    doSomething()  # false branch comment2\n" +
      "    # false branch comment3\n";

    // We only keep comments of the else branch
    String fixed = "doSomething()  # false branch comment2\n" +
      "    # false branch comment3\n";

    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
  }

  @Test
  void lambda(){
    String noncompliant = "a = (lambda x: x+1\n" +
      "     if x > 0 # Noncompliant\n" +
      "     else x+1)";
    String fixed = "a = (lambda x: x+1)";
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
  }

  @Test
  void multiple_conditional_statements(){
    String noncompliant = "a = 1 if x else 1 if y else 1 if z else 1";
    String fixed = "a = 1";
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);

    noncompliant = "a = (1 if x else 1) if cond else 1";
    fixed = "a = 1";
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
  }

  @Test
  void wrapped_conditional_expression(){
    PythonQuickFixVerifier.verify(check,
      "a = (1 if x else 1)",
      "a = (1)"
    );
  }

  @Test
  void test_elseif(){
    String noncompliant ="def func():\n" +
      "    if b == 0:\n" +
      "        doSomething()\n" +
      "    elif b == 1:\n" +
      "        doSomething()\n" +
      "    else:\n" +
      "        doSomething()\n";
    String fixed = "def func():\n" +
      "    doSomething()\n";
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
  }

  @Test
  void test_elseif_multiple(){
    String noncompliant ="def func():\n" +
      "    if b == 0:\n" +
      "        doSomething()\n" +
      "        doOneMoreThing()\n"+
      "    elif b == 1:\n" +
      "        doSomething()\n" +
      "        doOneMoreThing()\n"+
      "    else:\n" +
      "        doSomething()\n"+
      "        doOneMoreThing()\n";;
    String fixed = "def func():\n" +
      "    doSomething()\n"+
      "    doOneMoreThing()\n";;
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
  }

  @Test
  void test_elseif_more(){
    String noncompliant ="def func():\n" +
      "    if b == 0:\n" +
      "        doSomething()\n" +
      "        doSomething()\n" +
      "        doOneMoreThing()\n"+
      "    elif b == 1:\n" +
      "        doSomething()\n" +
      "        doSomething()\n" +
      "        doOneMoreThing()\n"+
      "    else:\n" +
      "        doSomething()\n"+
      "        doSomething()\n" +
      "        doOneMoreThing()\n";;
    String fixed = "def func():\n" +
      "    doSomething()\n"+
      "    doSomething()\n" +
      "    doOneMoreThing()\n";;
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
  }

  @Test
  void test_complex_condition(){
    String noncompliant = "a = do_something(a, b, c, do_something_else(d)) if x else do_something(a, b, c, do_something_else(d))";
    String fixed = "a = do_something(a, b, c, do_something_else(d))";
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
  }

  @Test
  void test_remove(){
    String noncompliant = ""+
      "if b == 0:\n" +
      "    doSomething()\n" +
      "else:\n" +
      "    doSomething()\n" +
      "\n" +
      "a = 1";
    String fixed = ""+
      "doSomething()\n" +
      "\n" +
      "a = 1";
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);

    noncompliant = ""+
      "if b == 0:\n" +
      "    doSomething()\n" +
      "else:\n" +
      "    doSomething()\n" +
      "\n" +
      "\n" +
      "a = 1";
    fixed = ""+
      "doSomething()\n" +
      "\n" +
      "\n" +
      "a = 1";
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);

    noncompliant = ""+
      "def foo():\n" +
      "    if a == b:\n" +
      "        doSomething()\n" +
      "    else:\n" +
      "        doSomething()\n" +
      "    doSomethingElse()\n";

    fixed = ""+
      "def foo():\n" +
      "    doSomething()\n" +
      "    doSomethingElse()\n";

    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
  }
}
