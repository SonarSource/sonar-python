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

class CaughtExceptionsCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/caughtExceptions.py", new CaughtExceptionsCheck());
  }

  @Test
  void quickFixTest() {
    var check = new CaughtExceptionsCheck();
    var before = """
      class CustomException:
        ...
      
      def foo():
        try:
          a = bar()
        except CustomException:
          print("Exception")""";

    var after = """
      class CustomException(Exception):
        ...
      
      def foo():
        try:
          a = bar()
        except CustomException:
          print("Exception")""";

    var expectedMessage = String.format(CaughtExceptionsCheck.QUICK_FIX_MESSAGE_FORMAT, "CustomException");

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, expectedMessage);
  }

  @Test
  void quickFixIPythonTest() {
    var check = new CaughtExceptionsCheck();
    var before = """
      #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER
      class CustomException:
        ...
      
      #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER
      def foo():
        try:
          a = %time bar()
        except CustomException:
          ...""";

    var after = """
      #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER
      class CustomException(Exception):
        ...
      
      #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER
      def foo():
        try:
          a = %time bar()
        except CustomException:
          ...""";

    var expectedMessage = String.format(CaughtExceptionsCheck.QUICK_FIX_MESSAGE_FORMAT, "CustomException");

    PythonQuickFixVerifier.verifyIPython(check, before, after);
    PythonQuickFixVerifier.verifyIPythonQuickFixMessages(check, before, expectedMessage);
  }

  @Test
  void exceptionWithEmptyParenthesisQuickFixTest() {
    var check = new CaughtExceptionsCheck();
    var before = """
      class CustomException():
        ...
      
      def foo():
        try:
          a = bar()
        except CustomException:
          print("Exception")""";

    var after = """
      class CustomException(Exception):
        ...
      
      def foo():
        try:
          a = bar()
        except CustomException:
          print("Exception")""";

    var expectedMessage = String.format(CaughtExceptionsCheck.QUICK_FIX_MESSAGE_FORMAT, "CustomException");

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, expectedMessage);
  }

  @Test
  void exceptionWithNotEmptyParenthesisQuickFixTest() {
    var check = new CaughtExceptionsCheck();
    var before = """
      class AbcEx:
          ...
      
      class Ex(AbcEx):
          ...
      
      def someLogic():
          try:
              a = foo()
          except Ex:
              ...""";

    var after = """
      class AbcEx:
          ...
      
      class Ex(AbcEx, Exception):
          ...
      
      def someLogic():
          try:
              a = foo()
          except Ex:
              ...""";

    var expectedMessage = String.format(CaughtExceptionsCheck.QUICK_FIX_MESSAGE_FORMAT, "Ex");

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, expectedMessage);
  }
}
