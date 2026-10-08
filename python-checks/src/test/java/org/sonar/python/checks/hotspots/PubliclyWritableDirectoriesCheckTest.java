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

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class PubliclyWritableDirectoriesCheckTest {
  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/hotspots/publiclyWritableDirectories.py", new PubliclyWritableDirectoriesCheck());
  }

  /**
   * Verifies that a project-local tempfile-like API does not receive the standard-library exemption.
   */
  @Test
  void reportsProjectLocalTempfileApi() {
    PythonCheckVerifier.verify(List.of(
      "src/test/resources/checks/hotspots/publiclyWritableDirectoriesLocalTempfile.py",
      "src/test/resources/checks/hotspots/tempfile.py"), new PubliclyWritableDirectoriesCheck());
  }

  @Test
  void recognizesOnlyExternalCallableSymbols() {
    FunctionSymbol externalFunction = mock(FunctionSymbol.class);
    ClassSymbol externalClass = mock(ClassSymbol.class);
    FunctionSymbol localFunction = mock(FunctionSymbol.class);
    ClassSymbol localClass = mock(ClassSymbol.class);
    LocationInFile localFunctionDefinition = mock(LocationInFile.class);
    LocationInFile localClassDefinition = mock(LocationInFile.class);
    when(localFunction.definitionLocation()).thenReturn(localFunctionDefinition);
    when(localClass.definitionLocation()).thenReturn(localClassDefinition);

    AmbiguousSymbol externalAlternatives = mock(AmbiguousSymbol.class);
    when(externalAlternatives.alternatives()).thenReturn(Set.of(externalFunction, externalClass));
    AmbiguousSymbol localAlternative = mock(AmbiguousSymbol.class);
    when(localAlternative.alternatives()).thenReturn(Set.of(externalFunction, localClass));

    assertThat(PubliclyWritableDirectoriesCheck.isExternalCallable(externalFunction)).isTrue();
    assertThat(PubliclyWritableDirectoriesCheck.isExternalCallable(externalClass)).isTrue();
    assertThat(PubliclyWritableDirectoriesCheck.isExternalCallable(localFunction)).isFalse();
    assertThat(PubliclyWritableDirectoriesCheck.isExternalCallable(localClass)).isFalse();
    assertThat(PubliclyWritableDirectoriesCheck.isExternalCallable(externalAlternatives)).isTrue();
    assertThat(PubliclyWritableDirectoriesCheck.isExternalCallable(localAlternative)).isFalse();
    assertThat(PubliclyWritableDirectoriesCheck.isExternalCallable(mock(Symbol.class))).isFalse();
  }
}
