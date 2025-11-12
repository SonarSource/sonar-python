/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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

class IgnoredSystemExitCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/ignoredSystemExit.py", new IgnoredSystemExitCheck());
  }

  @Test
  void quickFixTest() {
    var before = """
      try:
          open("foo.txt", "r")
      except SystemExit:
          pass""";
    var after = """
      try:
          open("foo.txt", "r")
      except SystemExit:
          raise""";
    IgnoredSystemExitCheck check = new IgnoredSystemExitCheck();
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, IgnoredSystemExitCheck.QUICK_FIX_MESSAGE);
  }

  @Test
  void addRaiseQuickFixTest() {
    var before = """
      def bar():
          try:
              foo()
          except SystemExit:
              pass
              a = 10
              print(a)""";
    var after = """
      def bar():
          try:
              foo()
          except SystemExit:
              pass
              a = 10
              print(a)
              raise""";
    IgnoredSystemExitCheck check = new IgnoredSystemExitCheck();
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, IgnoredSystemExitCheck.QUICK_FIX_MESSAGE);
  }

  @Test
  void replacePassWithRaiseQuickFixTest() {
    var before = """
      def bar():
          try:
              foo()
          except SystemExit:
              a = 10
              print(a)
              pass""";
    var after = """
      def bar():
          try:
              foo()
          except SystemExit:
              a = 10
              print(a)
              raise""";
    IgnoredSystemExitCheck check = new IgnoredSystemExitCheck();
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, IgnoredSystemExitCheck.QUICK_FIX_MESSAGE);
  }

  @Test
  void replaceEllipsisWithRaiseQuickFixTest() {
    var before = """
      def bar():
          try:
              foo()
          except SystemExit:
              a = 10
              print(a)
              ...""";
    var after = """
      def bar():
          try:
              foo()
          except SystemExit:
              a = 10
              print(a)
              raise""";
    IgnoredSystemExitCheck check = new IgnoredSystemExitCheck();
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, IgnoredSystemExitCheck.QUICK_FIX_MESSAGE);
  }

  @Test
  void baseExceptionQuickFixTest() {
    var before = """
      def bar():
          try:
              foo()
          except BaseException:
              a = 10
              print(a)
              ...""";
    var after = """
      def bar():
          try:
              foo()
          except BaseException:
              a = 10
              print(a)
              raise""";
    IgnoredSystemExitCheck check = new IgnoredSystemExitCheck();
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, IgnoredSystemExitCheck.QUICK_FIX_MESSAGE);
  }

  @Test
  void bareExceptionQuickFixTest() {
    var before = """
      def bar():
          try:
              foo()
          except:
              a = 10
              print(a)
              ...""";
    var after = """
      def bar():
          try:
              foo()
          except:
              a = 10
              print(a)
              raise""";
    IgnoredSystemExitCheck check = new IgnoredSystemExitCheck();
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, IgnoredSystemExitCheck.QUICK_FIX_MESSAGE);
  }

}
