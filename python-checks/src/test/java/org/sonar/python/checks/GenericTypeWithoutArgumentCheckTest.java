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
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.types.v2.GenericInfo.BareTypeArgumentStatus;
import org.sonar.plugins.python.api.types.v2.GenericInfo;
import org.sonar.python.SubscriptionVisitor;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.checks.utils.PythonCheckVerifier;
import org.sonar.python.index.GenericInfoProtobuf;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.tree.NameImpl;
import org.sonar.python.types.protobuf.SymbolsProtos;

import static org.assertj.core.api.Assertions.assertThat;

class GenericTypeWithoutArgumentCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/genericTypeWithoutArgument.py", new GenericTypeWithoutArgumentCheck());
  }

  @ParameterizedTest
  @MethodSource("genericStates")
  void generic_defaults_in_annotations(GenericInfo genericInfo, int expectedIssues) {
    var file = new TestPythonVisitorRunner.MockPythonFile("", "defaults.py", """
      value: Generic
      nested: list[Generic]
      def fn(arg: Generic) -> Generic:
          ...
      """);
    var context = TestPythonVisitorRunner.createContext(file, null, "", ProjectLevelSymbolTable.empty(), CacheContextImpl.dummyCache());
    var descriptor = SymbolsProtos.ClassSymbol.newBuilder().setName("Generic").setFullyQualifiedName("defaults.Generic")
      .setIsGeneric(true).setGenericInfo(GenericInfoProtobuf.toProtobuf(genericInfo)).build();
    var symbol = new ClassSymbolImpl(descriptor, "defaults");
    // Inject a serialized class independently of the bundled Typeshed revision.
    context.rootTree().accept(new BaseTreeVisitor() {
      @Override
      public void visitName(Name name) {
        if ("Generic".equals(name.name())) {
          ((NameImpl) name).setSymbol(symbol);
        }
      }
    });
    SubscriptionVisitor.analyze(List.of(new GenericTypeWithoutArgumentCheck()), context);
    assertThat(context.getIssues()).hasSize(expectedIssues);
  }

  static Stream<Arguments> genericStates() {
    return Stream.of(
      Arguments.of(GenericInfo.UNKNOWN, 0),
      Arguments.of(GenericInfo.NON_GENERIC, 0),
      Arguments.of(new GenericInfo.Generic(BareTypeArgumentStatus.UNKNOWN), 4),
      Arguments.of(new GenericInfo.Generic(BareTypeArgumentStatus.CONTAINS_ANY), 4),
      Arguments.of(new GenericInfo.Generic(BareTypeArgumentStatus.NO_ANY), 0));
  }

  @Test
  void concrete_defaults_survive_project_reexports() {
    PythonCheckVerifier.verify(List.of(
      "src/test/resources/checks/genericDefaultsExported.py",
      "src/test/resources/checks/genericDefaultsImporting.py"), new GenericTypeWithoutArgumentCheck());
  }

  @Test
  void test_cross_file() {
    PythonCheckVerifier.verify(
      List.of(
        "src/test/resources/checks/genericTypeWithoutArgumentImported.py",
        "src/test/resources/checks/genericTypeWithoutArgumentImporting.py"
      ),
      new GenericTypeWithoutArgumentCheck()
    );
  }
}
