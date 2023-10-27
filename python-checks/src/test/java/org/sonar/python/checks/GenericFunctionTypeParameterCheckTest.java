/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
 * mailto:info AT sonarsource DOT com
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
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.checks;

import java.util.EnumSet;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.assertj.core.api.Assertions.assertThat;

class GenericFunctionTypeParameterCheckTest {


  public static final GenericFunctionTypeParameterCheck CHECK = new GenericFunctionTypeParameterCheck();

  @Test
  void new_python_test() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_312));
    PythonCheckVerifier.verify("src/test/resources/checks/genericFunctionTypeParameter.py", CHECK);
  }

  @Test
  void old_python_test() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_311, PythonVersionUtils.Version.V_312));
    var issues = PythonCheckVerifier.issues("src/test/resources/checks/genericFunctionTypeParameter.py", CHECK);
    assertThat(issues)
      .isEmpty();
  }

  @Test
  void quickfix_test_generic_with_bound() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_312));
    final String non_compliant =
      "from typing import TypeVar\n" +
        "_T = TypeVar(\"_T\", bound=str)\n" +
        "def func(a: _T, b: int) -> str:\n" +
        "    ...";
    final String compliant =
      "from typing import TypeVar\n" +
        "def func[_T: str](a: _T, b: int) -> str:\n" +
        "    ...";
    PythonQuickFixVerifier.verify(CHECK, non_compliant, compliant);
  }

  @Test
  void quickfix_test_generic_used_twice() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_312));
    final String non_compliant =
      "from typing import TypeVar\n" +
        "_T = TypeVar(\"_T\", bound=str)\n" +
        "i = 42\n" +
        "def func(a: _T, b: int) -> _T:\n" +
        "    ...";
    final String compliant =
      "from typing import TypeVar\n" +
        "i = 42\n" +
        "def func[_T: str](a: _T, b: int) -> _T:\n" +
        "    ...";
    PythonQuickFixVerifier.verify(CHECK, non_compliant, compliant);
  }

  @Test
  void quickfix_test_generic_two_functions() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_312));
    final String non_compliant =
      "from typing import TypeVar\n" +
        "_T = TypeVar(\"_T\", bound=str)\n" +
        "def func_a(a: _T, b: int) -> str:\n" +
        "    ...\n" +
        "def func_b(a: _T, b: int) -> str:\n" +
        "    ...\n";
    final String compliant =
      "from typing import TypeVar\n" +
        "_T = TypeVar(\"_T\", bound=str)\n" +
        "def func_a[_T: str](a: _T, b: int) -> str:\n" +
        "    ...\n" +
        "def func_b(a: _T, b: int) -> str:\n" +
        "    ...\n";
    PythonQuickFixVerifier.verify(CHECK, non_compliant, compliant);
  }

  @Test
  void quickfix_test_three_generics() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_312));
    final String non_compliant =
      "from typing import TypeVar\n" +
        "_T = TypeVar(\"_T\", bound=str)\n" +
        "_R = TypeVar(\"_R\")\n" +
        "_S = TypeVar(\"_S\")\n" +
        "def func(a: _R, b: _S) -> _T:\n" +
        "    ...";

    final String compliant =
      "from typing import TypeVar\n" +
        "def func[_R, _S, _T: str](a: _R, b: _S) -> _T:\n" +
        "    ...";
    PythonQuickFixVerifier.verify(CHECK, non_compliant, compliant);
  }

  @Test
  void quickfix_test_mixed_use_of_generics() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_312));
    final String non_compliant =
      "from typing import TypeVar\n" +
        "_T = TypeVar(\"_T\", bound=str)\n" +
        "_S = TypeVar(\"_S\")\n" +
        "def func[_R](a: _T, b: _R) -> _S:" +
        "    ...";

    final String compliant =
      "from typing import TypeVar\n" +
        "def func[_R, _S, _T: str](a: _T, b: _R) -> _S:" +
        "    ...";
    PythonQuickFixVerifier.verify(CHECK, non_compliant, compliant);
  }

  @Test
  void quickfix_test_many_generics() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_312));
    final String non_compliant =
      "from typing import TypeVar\n" +
        "_Z = TypeVar(\"_Z\")\n" +
        "_A = TypeVar(\"_A\")\n" +
        "_T = TypeVar(\"_T\", \nbound=str)\n" +
        "_H = TypeVar(\"_H\", bound=float)\n" +
        "_S = TypeVar(\"_S\")\n" +
        "def func[_R, _K](a: _T, b: _R, c: _H, d: _Z, e: _A) -> _S:\n" +
        "    ...";

    final String compliant =
      "from typing import TypeVar\n" +
        "def func[_R, _K, _A, _H: float, _S, _T: str, _Z](a: _T, b: _R, c: _H, d: _Z, e: _A) -> _S:\n" +
        "    ...";
    PythonQuickFixVerifier.verify(CHECK, non_compliant, compliant);
  }

  @Test
  void quickfix_test_one_generic_with_multiple_assignment_one_without() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_312));
    final String non_compliant =
      "from typing import TypeVar\n" +
        "_T = __T = TypeVar(\"_T\", bound=str)\n" +
        "_R = TypeVar(\"_R\")\n" +
        "def func(a: _T, b: __T) -> _R:" +
        "    ...";

    final String compliant =
      "from typing import TypeVar\n" +
        "_T = __T = TypeVar(\"_T\", bound=str)\n" +
        "def func[_R](a: _T, b: __T) -> _R:" +
        "    ...";
    PythonQuickFixVerifier.verify(CHECK, non_compliant, compliant);
  }

  @Test
  void test_no_quickfix() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_312));
    final String non_compliant =
      "from typing import TypeVar\n" +
        "_T = __T = TypeVar(\"_T\", bound=str)\n" +
        "def func(a: _T, b: int) -> str:" +
        "    ...";
    PythonQuickFixVerifier.verifyNoQuickFixes(CHECK, non_compliant);
  }


}
