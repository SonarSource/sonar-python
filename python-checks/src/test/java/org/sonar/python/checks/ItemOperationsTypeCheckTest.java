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
package org.sonar.python.checks;

import java.util.EnumSet;
import java.util.List;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonVersionUtils.Version;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.python.checks.utils.PythonCheckVerifier;
import org.sonar.python.types.TypeShed;

import static org.assertj.core.api.Assertions.assertThat;

class ItemOperationsTypeCheckTest {

  @AfterEach
  void resetPythonVersion() {
    ProjectPythonVersion.setCurrentVersions(PythonVersionUtils.allVersions());
    TypeShed.resetBuiltinSymbols();
  }

  @ParameterizedTest
  @EnumSource(value = Version.class, names = {"V_310", "V_311", "V_312", "V_313", "V_314", "V_315"})
  void unionTypeSubscriptionRespectsPythonVersion(Version version) {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(version));
    TypeShed.resetBuiltinSymbols();
    var issues = PythonCheckVerifier.issues("src/test/resources/checks/itemOperationsTypeCheck/unionTypeSubscription.py", new ItemOperationsTypeCheck());
    assertThat(issues).hasSize(version.compare(3, 14) >= 0 ? 0 : 3);
  }

  @Test
  void unionTypeSubscriptionWithMixedVersions() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(Version.V_313, Version.V_314));
    TypeShed.resetBuiltinSymbols();
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/itemOperationsTypeCheck/unionTypeSubscription.py", new ItemOperationsTypeCheck());
  }

  @Test
  void test_delitem() {
    PythonCheckVerifier.verify("src/test/resources/checks/itemOperationsTypeCheck/itemOperations_delitem.py", new ItemOperationsTypeCheck());
  }

  @Test
  void test_getitem() {
    PythonCheckVerifier.verify(List.of(
      "src/test/resources/checks/itemOperationsTypeCheck/itemOperations_getitem.py",
      "src/test/resources/checks/itemOperationsTypeCheck/importedGeneric.py",
      "src/test/resources/checks/classDecorators/importedDecorators.py"), new ItemOperationsTypeCheck());
  }

  @Test
  void test_setitem() {
    PythonCheckVerifier.verify("src/test/resources/checks/itemOperationsTypeCheck/itemOperations_setitem.py", new ItemOperationsTypeCheck());
  }

  @Test
  void instance_decorator_effects_on_item_operations() {
    PythonCheckVerifier.verify(List.of(
      "src/test/resources/checks/classDecorators/instanceItems.py",
      "src/test/resources/checks/classDecorators/importedDecorators.py"), new ItemOperationsTypeCheck());
  }

  @Test
  void project_function_marker_remains_conservative() {
    PythonCheckVerifier.verifyNoIssue(List.of(
      "src/test/resources/checks/classDecoratorFunctionImpostor/usage.py",
      "src/test/resources/checks/classDecoratorFunctionImpostor/typing.py"), new ItemOperationsTypeCheck());
  }
}
