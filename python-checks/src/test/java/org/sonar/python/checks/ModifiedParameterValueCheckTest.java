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
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class ModifiedParameterValueCheckTest {

  private final PythonCheck check = new ModifiedParameterValueCheck();
  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/modifiedParameterValue.py", check);
  }

  @Test
  void list_modified_quickfix() {
    String codeWithIssue = """
      def list_modified(param=list()):
          param.append('a')""";

    String fixedCode = """
      def list_modified(param=None):
          if param is None:
              param = list()
          param.append('a')""";

    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
  }

  @Test
  void method_with_quickfix() {
    String codeWithIssue = """
      class Foo:
          def list_modified(param=list()):
              param.append('a')""";

    String fixedCode = """
      class Foo:
          def list_modified(param=None):
              if param is None:
                  param = list()
              param.append('a')""";

    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
  }

  @Test
  void default_with_spaces_quickfix() {
    String codeWithIssue = """
      def list_modified(param = list()):
          param.append('a')""";

    String fixedCode = """
      def list_modified(param = None):
          if param is None:
              param = list()
          param.append('a')""";

    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
  }

  @Test
  void annotated_parameter_quickfix() {
    String codeWithIssue = """
      def list_modified(param:list=list()):
          param.append('a')""";

    String fixedCode = """
      def list_modified(param:list=None):
          if param is None:
              param = list()
          param.append('a')""";

    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
  }

  @Test
  void annotated_parameter_with_space_quickfix() {
    String codeWithIssue = """
      def list_modified(param: list = list()):
          param.append('a')""";

    String fixedCode = """
      def list_modified(param: list = None):
          if param is None:
              param = list()
          param.append('a')""";

    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
  }

  @Test
  void set_modified_quickfix() {
    String codeWithIssue = """
      def set_modified(param=set()):
          param.add('a')""";

    String fixedCode = """
      def set_modified(param=None):
          if param is None:
              param = set()
          param.add('a')""";

    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
  }

  @Test
  void counter_modified_quickfix() {
    String codeWithIssue = """
      import collections
      def counter_modified(param=collections.Counter()):
          param.subtract()""";

    String fixedCode = """
      import collections
      def counter_modified(param=None):
          if param is None:
              param = collections.Counter()
          param.subtract()""";

    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
  }

  @Test
  void import_from_quickfix() {
    String codeWithIssue = """
      from collections import Counter
      def list_modified(param=Counter()):
          param.subtract()""";

    String fixedCode = """
      from collections import Counter
      def list_modified(param=None):
          if param is None:
              param = Counter()
          param.subtract()""";

    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
  }

  @Test
  void literal_dict_quickfix() {
    String codeWithIssue = """
      def literal_dict(param={}):
          param.pop('a')""";
    String fixedCode = """
      def literal_dict(param=None):
          if param is None:
              param = {}
          param.pop('a')""";

    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
  }

  @Test
  void literal_list_quickfix() {
    String codeWithIssue = """
      def literal_dict(param=[]):
          param.append('a')""";
    String fixedCode = """
      def literal_dict(param=None):
          if param is None:
              param = []
          param.append('a')""";

    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
  }

  @Test
  void set_quickfix() {
    String codeWithIssue = """
      def literal_dict(param=set()):
          param.add('a')""";
    String fixedCode = """
      def literal_dict(param=None):
          if param is None:
              param = set()
          param.add('a')""";

    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
  }

  @Test
  void no_quick_fix_multiline_assignment() {
    String codeWithIssue = """
      def no_quick_fix(a = [
        100
        ]):
        a.append(200)""";

    PythonQuickFixVerifier.verifyNoQuickFixes(check, codeWithIssue);
  }

  @Test
  void quickfix_non_empty_literal_dict() {
    String codeWithIssue = """
      def literal_dict(param={'foo': 'bar'}):
          param.pop('a')""";
    String fixedCode = """
      def literal_dict(param=None):
          if param is None:
              param = {'foo': 'bar'}
          param.pop('a')""";
    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
  }

  @Test
  void quickfix_non_empty_literal_list() {
    String codeWithIssue = """
      def literal_dict(param=[100, 200]):
          param.append('a')""";
    String fixedCode = """
      def literal_dict(param=None):
          if param is None:
              param = [100, 200]
          param.append('a')""";
    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
  }

  @Test
  void quickfix_non_empty_call() {
    String codeWithIssue = """
      def literal_dict(param=A('foo')):
          param.attr = 'bar'""";
    String fixedCode = """
      def literal_dict(param=None):
          if param is None:
              param = A('foo')
          param.attr = 'bar'""";
    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
  }

  @Test
  void no_quickfix_non_empty_set() {
    String codeWithIssue = """
      def literal_dict(param={'foo'}):
          param.attr = 'bar'""";
    String fixedCode = """
      def literal_dict(param=None):
          if param is None:
              param = {'foo'}
          param.attr = 'bar'""";
    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
  }

}
