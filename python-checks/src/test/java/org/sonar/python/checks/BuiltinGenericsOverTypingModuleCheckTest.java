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
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.sonar.python.checks.BuiltinGenericsOverTypingModuleCheck.MESSAGE;
import static org.sonar.python.checks.utils.CodeTestUtils.code;

class BuiltinGenericsOverTypingModuleCheckTest {

  @Test
  void checkBuiltins() {
    PythonCheckVerifier.verify("src/test/resources/checks/builtinGenericsOverTypingModule.py",
      new BuiltinGenericsOverTypingModuleCheck());
  }

  @Test
  void checkCollections() {
    PythonCheckVerifier.verify("src/test/resources/checks/builtinCollectionsOverTypingModule.py",
      new BuiltinGenericsOverTypingModuleCheck());
  }

  @Test
  void quick_fix_generics_in_param() {
    String codeWithIssue = code(
      "from typing import List",
      "def foo(test: List[int]) -> str:",
      "    return \"bar\"");
    String codeFixed = code(
      "from typing import List",
      "def foo(test: list[int]) -> str:",
      "    return \"bar\"");
    BuiltinGenericsOverTypingModuleCheck check = new BuiltinGenericsOverTypingModuleCheck();
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    String specificMessage = String.format(MESSAGE, "list");
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, specificMessage);
  }

  @Test
  void quick_fix_generics_in_return_type() {
    String codeWithIssue = code(
      "from typing import Dict",
      "def foo(test: list[int]) -> Dict[str, int]:",
      "    return {}");
    String codeFixed = code(
      "from typing import Dict",
      "def foo(test: list[int]) -> dict[str, int]:",
      "    return {}");
    BuiltinGenericsOverTypingModuleCheck check = new BuiltinGenericsOverTypingModuleCheck();
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    String specificMessage = String.format(MESSAGE, "dict");
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, specificMessage);
  }

  @Test
  void quick_fix_in_return_type_multiline() {
    String codeWithIssue = code(
      "from typing import Dict",
      "def foo(test: list[int]) -> Dict[str,",
      "    int]:",
      "    return {}");
    String codeFixed = code(
      "from typing import Dict",
      "def foo(test: list[int]) -> dict[str,",
      "    int]:",
      "    return {}");
    BuiltinGenericsOverTypingModuleCheck check = new BuiltinGenericsOverTypingModuleCheck();
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    String specificMessage = String.format(MESSAGE, "dict");
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, specificMessage);
  }

  @Test
  void quick_fix_generics_in_variable() {
    String codeWithIssue = code(
      "from typing import Set",
      "def foo():",
      "    my_var: Set[str]",
      "    return my_var");
    String codeFixed = code(
      "from typing import Set",
      "def foo():",
      "    my_var: set[str]",
      "    return my_var");
    BuiltinGenericsOverTypingModuleCheck check = new BuiltinGenericsOverTypingModuleCheck();
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    String specificMessage = String.format(MESSAGE, "set");
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, specificMessage);
  }

  @Test
  void quick_fix_fully_qualified() {
    String codeWithIssue = code(
      "import typing",
      "def foo():",
      "    my_var: typing.Set[int]",
      "    return my_var");
    String codeFixed = code(
      "import typing",
      "def foo():",
      "    my_var: set[int]",
      "    return my_var");
    BuiltinGenericsOverTypingModuleCheck check = new BuiltinGenericsOverTypingModuleCheck();
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    String specificMessage = String.format(MESSAGE, "set");
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, specificMessage);
  }

  @Test
  void no_quick_fix_for_change_requiring_import() {
    String codeWithIssue = code(
      "from typing import Iterable",
      "def foo(test: Iterable[int]) -> dict[str, int]:",
      "    return {}");
    BuiltinGenericsOverTypingModuleCheck check = new BuiltinGenericsOverTypingModuleCheck();
    PythonQuickFixVerifier.verifyNoQuickFixes(check, codeWithIssue);
  }

  @Test
  void quick_fix_nested_types() {
    String codeWithIssue = code(
      "from typing import Dict",
      "def foo() -> tuple[str, Dict[int]]:",
      "    return (\"foo\", list())");
    String codeFixed = code(
      "from typing import Dict",
      "def foo() -> tuple[str, dict[int]]:",
      "    return (\"foo\", list())");
    BuiltinGenericsOverTypingModuleCheck check = new BuiltinGenericsOverTypingModuleCheck();
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    String specificMessage = String.format(MESSAGE, "dict");
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, specificMessage);
  }

  @Test
  void quick_fix_nested_function() {
    String codeWithIssue = code(
      "from typing import Tuple",
      "def foo() -> tuple[str, int]:",
      "    def bar(test: Tuple[str, int]) -> str:",
      "        return \"bar\"",
      "    return (\"foo\", list())");
    String codeFixed = code(
      "from typing import Tuple",
      "def foo() -> tuple[str, int]:",
      "    def bar(test: tuple[str, int]) -> str:",
      "        return \"bar\"",
      "    return (\"foo\", list())");
    BuiltinGenericsOverTypingModuleCheck check = new BuiltinGenericsOverTypingModuleCheck();
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    String specificMessage = String.format(MESSAGE, "tuple");
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, specificMessage);
  }
}
