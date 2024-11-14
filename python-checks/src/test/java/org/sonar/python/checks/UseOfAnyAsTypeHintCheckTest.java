/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class UseOfAnyAsTypeHintCheckTest {
  @Test
  void useOfAny() {
    PythonCheckVerifier.verify("src/test/resources/checks/useOfAnyAsTypeHintCheck/useOfAnyAsTypeHint.py", new UseOfAnyAsTypeHintCheck());
  }

  @Test
  void useOfTypingAny() {
    PythonCheckVerifier.verify("src/test/resources/checks/useOfAnyAsTypeHintCheck/useOftypingAnyAsTypeHint.py",
      new UseOfAnyAsTypeHintCheck());
  }

  @Test
  void useOfUserDefinedTypeCalledAny() {
    PythonCheckVerifier.verify("src/test/resources/checks/useOfAnyAsTypeHintCheck/useOfUserDefinedTypeAnyAsTypeHint.py",
      new UseOfAnyAsTypeHintCheck());
  }

  @Test
  void useOfOverrideOrOverloadDecorator() {
    PythonCheckVerifier.verify(
      List.of(
        "src/test/resources/checks/useOfAnyAsTypeHintCheck/useOfOverrideOrOverloadDecorator.py",
        "src/test/resources/checks/useOfAnyAsTypeHintCheck/reexport_typing_overload_override.py"
      ),
      new UseOfAnyAsTypeHintCheck()
    );
  }

  @Test
  void useOfAnyImported() {
    PythonCheckVerifier.verify(
      List.of(
        "src/test/resources/checks/useOfAnyAsTypeHintCheck/useOfAnyAsTypeHintImported.py",
        "src/test/resources/checks/useOfAnyAsTypeHintCheck/useOfAnyAsTypeHintImporting.py"
      ),
      new UseOfAnyAsTypeHintCheck()
    );
  }
}
