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

import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class DeprecatedNumpyTypesCheckTest {

  private final DeprecatedNumpyTypesCheck check = new DeprecatedNumpyTypesCheck();

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/deprecatedNumpyTypesCheck.py", check);
  }

  @ParameterizedTest(name = "quickfix: {0}")
  @MethodSource("numpyTypeProvider")
  void quickFix(String numpyType, String parameter, String quickFixType) {
    String failure =
      String.format("import numpy as np\n" +
      "def foo():\n" +
      "    b = %s%s", numpyType, parameter);

    String fixed =
      String.format("import numpy as np\n" +
      "def foo():\n" +
      "    b = %s%s", quickFixType, parameter);

    PythonQuickFixVerifier.verify(check, failure, fixed);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, failure, String.format("Replace with %s.", quickFixType));
  }
  
  private static Stream<Arguments> numpyTypeProvider(){
    return Stream.of(
      Arguments.of("np.bool", "(True)", "bool"),
      Arguments.of("np.int", "(42)", "int"),
      Arguments.of("np.float", "(4.2)", "float"),
      Arguments.of("np.complex", "(-2.0, -1.2)", "complex"),
      Arguments.of("np.str", "(\"test\")", "str"),
      Arguments.of("np.object", "", "object"),
      Arguments.of("np.long", "(2)", "int"),
      Arguments.of("np.unicode", "(\"test\")", "str")
    );
  }
}
