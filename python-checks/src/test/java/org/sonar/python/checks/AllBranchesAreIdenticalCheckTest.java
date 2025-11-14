/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
      """
      def func():
          if b == 0:
              doSomething()
          else:
              doSomething()
      """;
    String fixed =
      """
      def func():
          doSomething()
      """;
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
  }

  @Test
  void quickfix_semicolons() {
    String noncompliant =
      """
      def func():
          if b == 0:
              doSomething(); doOneMoreThing()
          else:
              doSomething(); doOneMoreThing()
      """;
    String fixed =
      """
      def func():
          doSomething(); doOneMoreThing()
      """;
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
  }

  @Test
  void if_enclosed() {
    String noncompliant =
      """
      def func():
          if b == 0:
              if a == 1:
                  doSomething()
          else:
              if a == 1:
                  doSomething()
      """;
    String fixed =
      """
      def func():
          if a == 1:
              doSomething()
      """;
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
      """
      if foo():
          doSomething()
      else:
          doSomething()
      """
    );
  }

  @Test
  void no_quick_fix_with_side_effect_within_operator_right_hand() {
    PythonQuickFixVerifier.verifyNoQuickFixes(check,
      """
      if 1 == 2 and foo():
          doSomething()
      else:
          doSomething()
      """
    );
  }

  @Test
  void no_quick_fix_with_side_effect_within_operator_left_hand() {
    PythonQuickFixVerifier.verifyNoQuickFixes(check,
      """
      if foo() and 1 == 2:
          doSomething()
      else:
          doSomething()
      """
    );
  }

  @Test
  void no_quick_fix_with_side_effect_within_operator_parenthesis() {
    PythonQuickFixVerifier.verifyNoQuickFixes(check,
      """
      if 1 == 3 or (foo() and 1 == 2):
          doSomething()
      else:
          doSomething()
      """
    );
  }

  @Test
  void no_quick_fix_with_side_effect_in_elif_condition() {
    PythonQuickFixVerifier.verifyNoQuickFixes(check,
      """
      if b == 0:
          doSomething()
      elif bar():
          doSomething()
      else:
          doSomething()
      """
    );
  }

  @Test
  void test_multiple_statement(){
    String noncompliant ="""
      def func():
          if b == 0:
              doSomething()
              doOneMoreThing()
          else:
              doSomething()
              doOneMoreThing()
      """;
    String fixed = """
      def func():
          doSomething()
          doOneMoreThing()
      """;
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
  }

  @Test
  void comments() {
    String noncompliant = """
      if a == 0:
          # true branch comment1
          doSomething()  # true branch comment2
          # true branch comment3
      else:
          # false branch comment1
          doSomething()  # false branch comment2
          # false branch comment3
      """;

    // We only keep comments of the else branch
    String fixed = """
      doSomething()  # false branch comment2
          # false branch comment3
      """;

    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
  }

  @Test
  void lambda(){
    String noncompliant = """
      a = (lambda x: x+1
           if x > 0 # Noncompliant
           else x+1)""";
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
    String noncompliant ="""
      def func():
          if b == 0:
              doSomething()
          elif b == 1:
              doSomething()
          else:
              doSomething()
      """;
    String fixed = """
      def func():
          doSomething()
      """;
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
  }

  @Test
  void test_elseif_multiple(){
    String noncompliant ="""
      def func():
          if b == 0:
              doSomething()
              doOneMoreThing()
          elif b == 1:
              doSomething()
              doOneMoreThing()
          else:
              doSomething()
              doOneMoreThing()
      """;
    String fixed = """
      def func():
          doSomething()
          doOneMoreThing()
      """;
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
  }

  @Test
  void test_elseif_more(){
    String noncompliant ="""
      def func():
          if b == 0:
              doSomething()
              doSomething()
              doOneMoreThing()
          elif b == 1:
              doSomething()
              doSomething()
              doOneMoreThing()
          else:
              doSomething()
              doSomething()
              doOneMoreThing()
      """;
    String fixed = """
      def func():
          doSomething()
          doSomething()
          doOneMoreThing()
      """;
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
    String noncompliant = """
      if b == 0:
          doSomething()
      else:
          doSomething()
      
      a = 1""";
    String fixed = """
      doSomething()
      
      a = 1""";
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);

    noncompliant = """
      if b == 0:
          doSomething()
      else:
          doSomething()
      
      
      a = 1""";
    fixed = """
      doSomething()
      
      
      a = 1""";
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);

    noncompliant = """
      def foo():
          if a == b:
              doSomething()
          else:
              doSomething()
          doSomethingElse()
      """;

    fixed = """
      def foo():
          doSomething()
          doSomethingElse()
      """;

    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
  }
}
