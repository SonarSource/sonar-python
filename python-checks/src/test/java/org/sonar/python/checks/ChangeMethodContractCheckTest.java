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

import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ParameterV2;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.python.semantic.v2.FunctionTypeBuilder;
import org.sonar.python.semantic.v2.ClassTypeBuilder;

import static org.assertj.core.api.Assertions.assertThat;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class ChangeMethodContractCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify(
      List.of(
        "src/test/resources/checks/changeMethodContract/changeMethodContract.py",
        "src/test/resources/checks/changeMethodContract/changeMethodContractParent.py"
      ),
      new ChangeMethodContractCheck()
    );
  }
  @Test
  void resolved_contracts() {
    PythonCheckVerifier.verify("src/test/resources/checks/changeMethodContract/resolvedContracts.py", new ChangeMethodContractCheck());
  }

  @Test
  void ruling_regressions() {
    PythonCheckVerifier.verify("src/test/resources/checks/changeMethodContract/rulingRegressions.py", new ChangeMethodContractCheck());
  }

  @Test
  void optional_keyword_additions_and_renamed_positional_parameters() {
    PythonCheckVerifier.verify("src/test/resources/checks/changeMethodContract/reviewFixes.py", new ChangeMethodContractCheck());
  }

  @Test
  void overloaded_contracts_require_consistent_defaults_and_parameter_modes() {
    FunctionType required = contract(false, false);
    FunctionType same = contract(false, false);
    assertThat(ChangeMethodContractCheck.unambiguousContract(UnionType.or(required, same))).isPresent();
    assertThat(ChangeMethodContractCheck.unambiguousContract(UnionType.or(required, contract(true, false)))).isEmpty();
    assertThat(ChangeMethodContractCheck.unambiguousContract(UnionType.or(required, contract(false, true)))).isEmpty();
    assertThat(ChangeMethodContractCheck.unambiguousContract(UnionType.or(required, PythonType.UNKNOWN))).isEmpty();
    var mixed = UnionType.or(required, new ClassTypeBuilder("NotAFunction", "mod.NotAFunction").build());
    assertThat(mixed).isInstanceOf(UnionType.class);
    assertThat(ChangeMethodContractCheck.unambiguousContract(mixed)).isEmpty();
  }

  @ParameterizedTest(name = "{0}")
  @MethodSource("ambiguousContracts")
  void overloaded_contracts_require_consistent_binding_and_signature_metadata(String description, FunctionType first, FunctionType other) {
    PythonType alternatives = UnionType.or(first, other);
    assertThat(alternatives).isInstanceOf(UnionType.class);
    assertThat(ChangeMethodContractCheck.unambiguousContract(alternatives)).isEmpty();
  }

  private static Stream<Arguments> ambiguousContracts() {
    FunctionType required = contract(false, false);
    FunctionType firstDecorator = new FunctionTypeBuilder("firstDecorator").build();
    FunctionType otherDecorator = new FunctionTypeBuilder("otherDecorator").build();
    return Stream.of(
      Arguments.of("variadic parameter", required, contractBuilder(false, false)
        .withHasVariadicParameter(true)
        .withParameters(List.of(new ParameterV2("value", TypeWrapper.UNKNOWN_TYPE_WRAPPER, false, false, false, false, true, null))).build()),
      Arguments.of("parameter count", required, contractBuilder(false, false).withParameters(List.of()).build()),
      Arguments.of("instance versus static method", required, contractBuilder(false, false).withInstanceMethod(true).build()),
      Arguments.of("class versus static method", required, contractBuilder(false, false).withClassMethod(true).build()),
      Arguments.of("unknown decorator metadata", required, contractBuilder(false, false).withHasDecorators(true).build()),
      Arguments.of("decorator identity",
        contractBuilder(false, false).withHasDecorators(true).withDecorators(List.of(TypeWrapper.of(firstDecorator))).build(),
        contractBuilder(false, false).withHasDecorators(true).withDecorators(List.of(TypeWrapper.of(otherDecorator))).build()),
      Arguments.of("parameter name", required, contractBuilder(false, false)
        .withParameters(List.of(new ParameterV2("other", TypeWrapper.UNKNOWN_TYPE_WRAPPER, false, false, false, false, false, null))).build()),
      Arguments.of("positional-only parameter", required, contractBuilder(false, false)
        .withParameters(List.of(new ParameterV2("value", TypeWrapper.UNKNOWN_TYPE_WRAPPER, false, false, true, false, false, null))).build()));
  }

  private static FunctionType contract(boolean hasDefault, boolean keywordOnly) {
    return contractBuilder(hasDefault, keywordOnly).build();
  }

  private static FunctionTypeBuilder contractBuilder(boolean hasDefault, boolean keywordOnly) {
    return new FunctionTypeBuilder("run")
      .withAttributes(List.of())
      .withDecorators(List.of())
      .withParameters(List.of(new ParameterV2("value", TypeWrapper.UNKNOWN_TYPE_WRAPPER,
        hasDefault, keywordOnly, false, false, false, null)));
  }

}
