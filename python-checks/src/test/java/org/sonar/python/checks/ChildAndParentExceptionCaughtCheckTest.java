/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class ChildAndParentExceptionCaughtCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/childAndParentExceptionCaughtCheck.py", new ChildAndParentExceptionCaughtCheck());
  }

  @Test
  void childWithParentQuickFixTest() {
    ChildAndParentExceptionCaughtCheck check = new ChildAndParentExceptionCaughtCheck();

    String before = """
      def child_with_parent():
        try:
            raise NotImplementedError()
        except (RuntimeError, RecursionError):
            print("Foo")""";
    String after = """
      def child_with_parent():
        try:
            raise NotImplementedError()
        except RuntimeError:
            print("Foo")""";
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, ChildAndParentExceptionCaughtCheck.QUICK_FIX_MESSAGE);
  }

  @Test
  void parentWithChildQuickFixTest() {
    ChildAndParentExceptionCaughtCheck check = new ChildAndParentExceptionCaughtCheck();

    String before = """
      def parent_with_child():
        try:
            raise NotImplementedError()
        except (RecursionError, RuntimeError):
            print("Foo")""";
    String after = """
      def parent_with_child():
        try:
            raise NotImplementedError()
        except RuntimeError:
            print("Foo")""";
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, ChildAndParentExceptionCaughtCheck.QUICK_FIX_MESSAGE);
  }

  @Test
  void duplicateExceptionQuickFixTest() {
    ChildAndParentExceptionCaughtCheck check = new ChildAndParentExceptionCaughtCheck();

    String before = """
      def duplicate_exception_caught():
        try:
            raise NotImplementedError()
        except (RuntimeError, RuntimeError):
            print("Foo")""";
    String after = """
      def duplicate_exception_caught():
        try:
            raise NotImplementedError()
        except RuntimeError:
            print("Foo")""";
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, ChildAndParentExceptionCaughtCheck.QUICK_FIX_MESSAGE);
  }

  @Test
  void threeExceptionsFirstQuickFixTest() {
    ChildAndParentExceptionCaughtCheck check = new ChildAndParentExceptionCaughtCheck();

    String before = """
      def child_with_parent():
        try:
            raise NotImplementedError()
        except (RecursionError, RuntimeError, Abc):
            print("Foo")""";
    String after = """
      def child_with_parent():
        try:
            raise NotImplementedError()
        except (RuntimeError, Abc):
            print("Foo")""";
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, ChildAndParentExceptionCaughtCheck.QUICK_FIX_MESSAGE);
  }

  @Test
  void threeExceptionsSecondQuickFixTest() {
    ChildAndParentExceptionCaughtCheck check = new ChildAndParentExceptionCaughtCheck();

    String before = """
      def child_with_parent():
        try:
            raise NotImplementedError()
        except (RuntimeError, RecursionError, Abc):
            print("Foo")""";
    String after = """
      def child_with_parent():
        try:
            raise NotImplementedError()
        except (RuntimeError, Abc):
            print("Foo")""";
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, ChildAndParentExceptionCaughtCheck.QUICK_FIX_MESSAGE);
  }

  @Test
  void threeExceptionsThirdQuickFixTest() {
    ChildAndParentExceptionCaughtCheck check = new ChildAndParentExceptionCaughtCheck();

    String before = """
      def child_with_parent():
        try:
            raise NotImplementedError()
        except (RuntimeError, Abc, RecursionError):
            print("Foo")""";
    String after = """
      def child_with_parent():
        try:
            raise NotImplementedError()
        except (RuntimeError, Abc):
            print("Foo")""";
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, ChildAndParentExceptionCaughtCheck.QUICK_FIX_MESSAGE);
  }

  @Test
  void threeExceptionsThirdWithCommaInTheEndQuickFixTest() {
    ChildAndParentExceptionCaughtCheck check = new ChildAndParentExceptionCaughtCheck();

    String before = """
      def child_with_parent():
        try:
            raise NotImplementedError()
        except (RuntimeError, Abc, RecursionError,):
            print("Foo")""";
    String after = """
      def child_with_parent():
        try:
            raise NotImplementedError()
        except (RuntimeError, Abc):
            print("Foo")""";
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, ChildAndParentExceptionCaughtCheck.QUICK_FIX_MESSAGE);
  }

  @Test
  void multiLineQuickFixTest() {
    ChildAndParentExceptionCaughtCheck check = new ChildAndParentExceptionCaughtCheck();

    String before = """
      def milty_line():
        try:
          raise NotImplementedError()
        except (RuntimeError,
          Abc, RecursionError,):
            print("Foo")""";
    String after = """
      def milty_line():
        try:
          raise NotImplementedError()
        except (RuntimeError, Abc):
            print("Foo")""";
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, ChildAndParentExceptionCaughtCheck.QUICK_FIX_MESSAGE);
  }

  @Test
  void multiLineTwoQuickFixTest() {
    ChildAndParentExceptionCaughtCheck check = new ChildAndParentExceptionCaughtCheck();

    String before = """
      def milty_line():
        try:
          raise NotImplementedError()
        except (RuntimeError,
          RecursionError,):
            print("Foo")""";
    String after = """
      def milty_line():
        try:
          raise NotImplementedError()
        except RuntimeError:
            print("Foo")""";
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, ChildAndParentExceptionCaughtCheck.QUICK_FIX_MESSAGE);
  }

  @Test
  void qualifiedExpressionExpressionFixTest() {
    ChildAndParentExceptionCaughtCheck check = new ChildAndParentExceptionCaughtCheck();

    String before = """
      def child_with_parent():
        try:
            raise NotImplementedError()
        except (RuntimeError, RecursionError, asd.Abc):
            print("Foo")""";
    String after = """
      def child_with_parent():
        try:
            raise NotImplementedError()
        except (RuntimeError, asd.Abc):
            print("Foo")""";
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, ChildAndParentExceptionCaughtCheck.QUICK_FIX_MESSAGE);
  }


  @Test
  void functionCallExpressionFixTest() {
    ChildAndParentExceptionCaughtCheck check = new ChildAndParentExceptionCaughtCheck();

    String before = """
      def child_with_parent():
        try:
            raise NotImplementedError()
        except (RuntimeError, RecursionError, asd.Abc()):
            print("Foo")""";
    PythonQuickFixVerifier.verifyNoQuickFixes(check, before);
  }
}
