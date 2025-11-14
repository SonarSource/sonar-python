/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class EmptyCollectionConstructorCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/emptyCollectionConstructor.py", new EmptyCollectionConstructorCheck());
  }

  private static Stream<Arguments> testQuickFixListConstructorSource() {
    return Stream.of(
      Arguments.of("dict()", "{}"),
      Arguments.of("tuple()", "()"),
      Arguments.of("list()", "[]")
    );
  }

  @ParameterizedTest
  @MethodSource("testQuickFixListConstructorSource")
  void testQuickFixListConstructor(String before, String after) {
    PythonQuickFixVerifier.verify(
      new EmptyCollectionConstructorCheck(),
      before,
      after
    );

    PythonQuickFixVerifier.verifyQuickFixMessages(new EmptyCollectionConstructorCheck(), before, "Replace with literal");
  }

  @Test
  void testNoQuickFix() {
    PythonQuickFixVerifier.verifyNoQuickFixes(new EmptyCollectionConstructorCheck(), "dict(test=1)");
  }

}
