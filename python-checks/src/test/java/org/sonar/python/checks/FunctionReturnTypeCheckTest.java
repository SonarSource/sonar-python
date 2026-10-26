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

import java.io.File;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeOrigin;
import org.sonar.python.SubscriptionVisitor;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.checks.utils.PythonCheckVerifier;
import org.sonar.python.semantic.v2.FunctionTypeBuilder;
import org.sonar.python.tree.CallExpressionImpl;
import org.sonar.python.tree.NameImpl;

import static org.assertj.core.api.Assertions.assertThat;

class FunctionReturnTypeCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/functionReturnType.py", new FunctionReturnTypeCheck());
  }

  @Test
  void library_annotations_allow_subclass_returns() {
    PythonCheckVerifier.verify("src/test/resources/checks/functionReturnTypeStubFactory.py", new FunctionReturnTypeCheck());
  }

  @ParameterizedTest
  @ValueSource(strings = {"stub", "local-callee", "unknown-return", "unknown-return-class"})
  void subclass_exception_requires_resolved_stub_types(String scenario) {
    var context = TestPythonVisitorRunner.createContext(new File("src/test/resources/checks/functionReturnTypeStubDisagreement.py"));
    // Preserve the inferred legacy mismatch and vary only the V2 evidence used by the exception.
    // These defensive cases cannot be exercised by local_factory: its legacy return type is unknown.
    context.rootTree().accept(new BaseTreeVisitor() {
      @Override
      public void visitCallExpression(CallExpression call) {
        switch (scenario) {
          case "local-callee" -> ((NameImpl) call.callee()).typeV2(new FunctionTypeBuilder("getLogger").withTypeOrigin(TypeOrigin.LOCAL).build());
          case "unknown-return" -> ((CallExpressionImpl) call).typeV2(PythonType.UNKNOWN);
          case "unknown-return-class" -> ((CallExpressionImpl) call).typeV2(ObjectType.fromType(PythonType.UNKNOWN));
          default -> { /* Keep the resolved stub as a positive control. */ }
        }
      }
    });

    SubscriptionVisitor.analyze(List.of(new FunctionReturnTypeCheck()), context);

    assertThat(context.getIssues()).hasSize("stub".equals(scenario) ? 0 : 1);
  }

  @Test
  void additionalTest() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/functionReturnTypeAdditionalTests.py", new FunctionReturnTypeCheck());
  }

  @Test
  void importedTypesTest() {
    PythonCheckVerifier.verify(List.of(
      "src/test/resources/checks/functionReturnTypeImported.py",
      "src/test/resources/checks/functionReturnTypeImporting.py"
    ), new FunctionReturnTypeCheck());
  }
  
}
