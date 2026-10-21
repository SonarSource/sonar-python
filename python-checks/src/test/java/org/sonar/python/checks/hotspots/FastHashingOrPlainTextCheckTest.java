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
package org.sonar.python.checks.hotspots;

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class FastHashingOrPlainTextCheckTest {
  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/fastHashingOrPlainText/fastHashingOrPlainText.py", new FastHashingOrPlainTextCheck());
  }

  @Test
  void cheapestAnywhereImportFrom() {
    PythonCheckVerifier.verify("src/test/resources/checks/fastHashingOrPlainText/fastHashingOrPlainTextArgon2ImportFrom.py",
      new FastHashingOrPlainTextCheck());
  }

  @Test
  void cheapestAnywhereImport() {
    PythonCheckVerifier.verify("src/test/resources/checks/fastHashingOrPlainText/fastHashingOrPlainTextArgon2Import.py",
      new FastHashingOrPlainTextCheck());
  }

  @Test
  void passlibArgon2Imports() {
    PythonCheckVerifier.verify("src/test/resources/checks/fastHashingOrPlainText/passlibArgon2Imports.py",
      new FastHashingOrPlainTextCheck());
  }

  @Test
  void passlibArgon2ImplementationWithoutPublicImport() {
    PythonCheckVerifier.verify("src/test/resources/checks/fastHashingOrPlainText/passlibArgon2Implementation.py",
      new FastHashingOrPlainTextCheck());
  }

  @Test
  void djangoSettings() {
    PythonCheckVerifier.verify("src/test/resources/checks/fastHashingOrPlainText/settings.py", new FastHashingOrPlainTextCheck());
  }

}
