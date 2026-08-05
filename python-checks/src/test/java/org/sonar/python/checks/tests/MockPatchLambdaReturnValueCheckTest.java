/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks.tests;

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.assertj.core.api.Assertions.assertThat;

class MockPatchLambdaReturnValueCheckTest {

  @Test
  void scope() {
    assertThat(new MockPatchLambdaReturnValueCheck().scope()).isEqualTo(PythonCheck.CheckScope.ALL);
  }

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/tests/mockPatchLambdaReturnValue.py", new MockPatchLambdaReturnValueCheck());
  }

  @Test
  void quickFixOnPositionalLambda() {
    PythonQuickFixVerifier.verify(new MockPatchLambdaReturnValueCheck(),
      """
        from unittest.mock import patch
        patch("app.api.fetch", lambda: {"ok": True})
        """,
      """
        from unittest.mock import patch
        patch("app.api.fetch", return_value={"ok": True})
        """);
  }

  @Test
  void quickFixOnKeywordLambda() {
    PythonQuickFixVerifier.verify(new MockPatchLambdaReturnValueCheck(),
      """
        from unittest.mock import patch
        patch("app.api.fetch", new=lambda *args: 42)
        """,
      """
        from unittest.mock import patch
        patch("app.api.fetch", return_value=42)
        """);
  }

  @Test
  void quickFixOnPatchObject() {
    PythonQuickFixVerifier.verify(new MockPatchLambdaReturnValueCheck(),
      """
        from unittest.mock import patch
        patch.object(Target, "attribute", lambda x, y: 7)
        """,
      """
        from unittest.mock import patch
        patch.object(Target, "attribute", return_value=7)
        """);
  }

  @Test
  void quickFixOnMockerPatch() {
    PythonQuickFixVerifier.verify(new MockPatchLambdaReturnValueCheck(),
      """
        def test_something(mocker):
            mocker.patch("module.target", (lambda: 7))
        """,
      """
        def test_something(mocker):
            mocker.patch("module.target", return_value=7)
        """);
  }

  @Test
  void quickFixMessage() {
    PythonQuickFixVerifier.verifyQuickFixMessages(new MockPatchLambdaReturnValueCheck(),
      """
        from unittest.mock import patch
        patch("app.api.fetch", lambda: 42)
        """,
      "Replace this lambda with a \"return_value\" argument");
  }

  @Test
  void noQuickFixWhenPositionalArgumentsFollow() {
    PythonQuickFixVerifier.verifyNoQuickFixes(new MockPatchLambdaReturnValueCheck(),
      """
        from unittest.mock import patch
        patch("app.api.fetch", lambda: 42, True)
        """);
  }

  @Test
  void quickFixWithKeywordArgumentAfterLambda() {
    PythonQuickFixVerifier.verify(new MockPatchLambdaReturnValueCheck(),
      """
        from unittest.mock import patch
        patch("app.api.fetch", lambda: 42, autospec=True)
        """,
      """
        from unittest.mock import patch
        patch("app.api.fetch", return_value=42, autospec=True)
        """);
  }

  @Test
  void noQuickFixOnDecorator() {
    PythonQuickFixVerifier.verifyNoQuickFixes(new MockPatchLambdaReturnValueCheck(),
      """
        from unittest.mock import patch
        @patch("app.api.fetch", lambda: 42)
        def test_something():
            ...
        """);
  }

  @Test
  void noQuickFixWhenLambdaBodyContainsCall() {
    PythonQuickFixVerifier.verifyNoQuickFixes(new MockPatchLambdaReturnValueCheck(),
      """
        from unittest.mock import patch
        def helper():
            ...
        with patch("app.api.fetch", lambda: helper()):
            ...
        """);
  }

  @Test
  void noQuickFixOnMultilineLambdaBody() {
    PythonQuickFixVerifier.verifyNoQuickFixes(new MockPatchLambdaReturnValueCheck(),
      """
        from unittest.mock import patch
        patch("app.api.fetch", lambda: {
            "ok": True,
        })
        """);
  }
}
