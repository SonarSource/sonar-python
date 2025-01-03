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

import static org.sonar.python.checks.utils.CodeTestUtils.code;

class MandatoryFunctionReturnTypeHintCheckTest {
  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/mandatoryFunctionReturnType.py", new MandatoryFunctionReturnTypeHintCheck());
  }

  @Test
  void quick_fix_none_return_type_for_init() {
    String codeWithIssue = code(
      "class A():",
      "    def __init__(self):",
      "        pass");
    String codeFixed = code(
      "class A():",
      "    def __init__(self) -> None:",
      "        pass");
    MandatoryFunctionReturnTypeHintCheck check = new MandatoryFunctionReturnTypeHintCheck();
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, MandatoryFunctionReturnTypeHintCheck.CONSTRUCTOR_MESSAGE);
  }

  @Test
  void quick_fix_none_return_type_for_init_weird_format() {
    String codeWithIssue = code(
      "class A():",
      "    def __init__(self)   : ",
      "        pass");
    String codeFixed = code(
      "class A():",
      "    def __init__(self) -> None   : ",
      "        pass");
    MandatoryFunctionReturnTypeHintCheck check = new MandatoryFunctionReturnTypeHintCheck();
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, MandatoryFunctionReturnTypeHintCheck.CONSTRUCTOR_MESSAGE);
  }

  @Test
  void quick_fix_none_return_type_for_init_multiline() {
    String codeWithIssue = code(
      "class A():",
      "    def __init__(self,",
      "        alice, bob):",
      "        pass");
    String codeFixed = code(
      "class A():",
      "    def __init__(self,",
      "        alice, bob) -> None:",
      "        pass");
    MandatoryFunctionReturnTypeHintCheck check = new MandatoryFunctionReturnTypeHintCheck();
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, MandatoryFunctionReturnTypeHintCheck.CONSTRUCTOR_MESSAGE);
  }

  @Test
  void quick_fix_add_return_type_literal() {
    String codeWithIssue = code(
      "def foo():",
      "    return \"bar\"");
    String codeFixed = code(
      "def foo() -> str:",
      "    return \"bar\"");
    MandatoryFunctionReturnTypeHintCheck check = new MandatoryFunctionReturnTypeHintCheck();
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, MandatoryFunctionReturnTypeHintCheck.MESSAGE);
  }

  @Test
  void quick_fix_return_expression() {
    String codeWithIssue = code(
      "def foo(alice: str):",
      "    return  alice + \"bar\"");
    String codeFixed = code(
      "def foo(alice: str) -> str:",
      "    return  alice + \"bar\"");
    MandatoryFunctionReturnTypeHintCheck check = new MandatoryFunctionReturnTypeHintCheck();
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, MandatoryFunctionReturnTypeHintCheck.MESSAGE);
  }

  @Test
  void quick_fix_no_return_keyword() {
    String codeWithIssue = code(
      "def foo(bob):",
      "    print(bob) ");
    String codeFixed = code(
      "def foo(bob) -> None:",
      "    print(bob) ");
    MandatoryFunctionReturnTypeHintCheck check = new MandatoryFunctionReturnTypeHintCheck();
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, MandatoryFunctionReturnTypeHintCheck.MESSAGE);
  }

  @Test
  void quick_fix_nested_return_type() {
    String codeWithIssue = code(
      "def foo(bob: bool):",
      "    if bob:",
      "        return 42",
      "    else:",
      "        return 40");
    String codeFixed = code(
      "def foo(bob: bool) -> int:",
      "    if bob:",
      "        return 42",
      "    else:",
      "        return 40");
    MandatoryFunctionReturnTypeHintCheck check = new MandatoryFunctionReturnTypeHintCheck();
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, MandatoryFunctionReturnTypeHintCheck.MESSAGE);
  }

  @Test
  void quick_fix_for_nested_function() {
    String codeWithIssue = code(
      "def foo() -> int:",
      "    def bar():",
      "        return \"foobar\"",
      "    return bar + 1");
    String codeFixed = code(
      "def foo() -> int:",
      "    def bar() -> str:",
      "        return \"foobar\"",
      "    return bar + 1");
    MandatoryFunctionReturnTypeHintCheck check = new MandatoryFunctionReturnTypeHintCheck();
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, MandatoryFunctionReturnTypeHintCheck.MESSAGE);
  }

  @Test
  void quick_fix_for_return_with_yield() {
    String codeWithIssue = code(
      "def foo(alice: dict[str, int]):",
      "    bar = 0",
      "    for k, v in alice.items():",
      "        _ = yield 41",
      "    return bar + 1");
    String codeFixed = code(
      "def foo(alice: dict[str, int]) -> int:",
      "    bar = 0",
      "    for k, v in alice.items():",
      "        _ = yield 41",
      "    return bar + 1");
    MandatoryFunctionReturnTypeHintCheck check = new MandatoryFunctionReturnTypeHintCheck();
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, MandatoryFunctionReturnTypeHintCheck.MESSAGE);
  }

  @Test
  void quick_fix_for_complex() {
    String codeWithIssue = code(
      "def foo():",
      "    return complex(1,2)");
    String codeFixed = code(
      "def foo() -> complex:",
      "    return complex(1,2)");
    MandatoryFunctionReturnTypeHintCheck check = new MandatoryFunctionReturnTypeHintCheck();
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, MandatoryFunctionReturnTypeHintCheck.MESSAGE);
  }

  @Test
  void quick_fix_for_float() {
    String codeWithIssue = code(
      "def foo():",
      "    return 3.14");
    String codeFixed = code(
      "def foo() -> float:",
      "    return 3.14");
    MandatoryFunctionReturnTypeHintCheck check = new MandatoryFunctionReturnTypeHintCheck();
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, MandatoryFunctionReturnTypeHintCheck.MESSAGE);
  }

  @Test
  void quick_fix_for_return_none() {
    String codeWithIssue = code(
      "def foo():",
      "    return None");
    String codeFixed = code(
      "def foo() -> None:",
      "    return None");
    MandatoryFunctionReturnTypeHintCheck check = new MandatoryFunctionReturnTypeHintCheck();
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, MandatoryFunctionReturnTypeHintCheck.MESSAGE);
  }

  @Test
  void no_quick_fix_collections() {
    String codeWithIssue = code(
      "from typing import Dict",
      "def foo(alice: Dict[str, int]):",
      "    bob = {}",
      "    for k, v in alice.items():",
      "        bob[k] = str(v)",
      "    return bob");
    MandatoryFunctionReturnTypeHintCheck check = new MandatoryFunctionReturnTypeHintCheck();
    PythonQuickFixVerifier.verifyNoQuickFixes(check, codeWithIssue);
  }

  @Test
  void no_quick_fix_tuples() {
    String codeWithIssue = code(
      "def foo():",
      "    return (\"bar\", 1)");
    MandatoryFunctionReturnTypeHintCheck check = new MandatoryFunctionReturnTypeHintCheck();
    PythonQuickFixVerifier.verifyNoQuickFixes(check, codeWithIssue);
  }

  @Test
  void no_quick_fix_generic_types() {
    String codeWithIssue = code(
      "def foo(alice: dict[str, int]):",
      "    bob = {}",
      "    for k, v in alice.items():",
      "        bob[k] = str(v)",
      "    return bob");
    MandatoryFunctionReturnTypeHintCheck check = new MandatoryFunctionReturnTypeHintCheck();
    PythonQuickFixVerifier.verifyNoQuickFixes(check, codeWithIssue);
  }

  @Test
  void no_quick_fix_ambiguous_type() {
    String codeWithIssue = code(
      "def foo(alice: str, bob: int):",
      "    return alice + bob");
    MandatoryFunctionReturnTypeHintCheck check = new MandatoryFunctionReturnTypeHintCheck();
    PythonQuickFixVerifier.verifyNoQuickFixes(check, codeWithIssue);
  }

  @Test
  void no_quick_fix_any_type() {
    String codeWithIssue = code(
      "def foo(alice: Any):",
      "    return alice");
    MandatoryFunctionReturnTypeHintCheck check = new MandatoryFunctionReturnTypeHintCheck();
    PythonQuickFixVerifier.verifyNoQuickFixes(check, codeWithIssue);
  }

  @Test
  void no_quick_fix_too_many_types() {
    String codeWithIssue = code(
      "def foo(bob: str):",
      "    if bob == \"foo\":",
      "        return \"life\" ",
      "    elif bob == \"bar\":",
      "        return 42",
      "    else:",
      "        return list()");
    MandatoryFunctionReturnTypeHintCheck check = new MandatoryFunctionReturnTypeHintCheck();
    PythonQuickFixVerifier.verifyNoQuickFixes(check, codeWithIssue);
  }

  @Test
  void no_quick_fix_for_yield() {
    String codeWithIssue = code(
      "def foo(alice: dict[str, int]):",
      "    for k, v in alice.items():",
      "        yield 42");
    MandatoryFunctionReturnTypeHintCheck check = new MandatoryFunctionReturnTypeHintCheck();
    PythonQuickFixVerifier.verifyNoQuickFixes(check, codeWithIssue);
  }

}
