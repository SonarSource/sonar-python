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
package org.sonar.python.semantic.v2.types.typecalculator;

import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.v2.FunctionTypeBuilder;
import org.sonar.python.types.v2.matchers.TypePredicateContext;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.sonar.python.types.v2.TypesTestUtils.INT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE;
import static org.sonar.python.types.v2.TypesTestUtils.parseAndInferTypes;

class QualifiedExpressionCalculatorTest {

  private final TypePredicateContext typePredicateContext = TypePredicateContext.of(PROJECT_LEVEL_TYPE_TABLE);
  private final QualifiedExpressionCalculator calculator = new QualifiedExpressionCalculator(typePredicateContext);

  @Test
  void nonReturningPropertyDoesNotExposeNeverAsAnExpressionType() {
    assertThat(propertyResult(PythonType.NEVER)).isSameAs(PythonType.UNKNOWN);
    var integer = ObjectType.fromType(INT_TYPE);
    assertThat(propertyResult(UnionType.or(PythonType.NEVER, integer))).isEqualTo(integer);
    assertThat(propertyResult(UnionType.or(PythonType.NEVER, PythonType.UNKNOWN))).isSameAs(PythonType.UNKNOWN);
  }

  @Test
  void nonReturningStubPropertyRemainsUnknown() {
    var fileInput = parseAndInferTypes("""
      import serial
      port = serial.VTIMESerial()
      port.cancel_read
      """);
    assertThat(getQualifiedExpressionFromStatement(fileInput).typeV2()).isSameAs(PythonType.UNKNOWN);
  }

  private PythonType propertyResult(PythonType returnType) {
    var function = new FunctionTypeBuilder("value").withReturnType(returnType)
      .withDecorators(List.of(TypeWrapper.of(PROJECT_LEVEL_TYPE_TABLE.getType("property")))).build();
    var owner = mock(PythonType.class);
    when(owner.resolveMember("value")).thenReturn(Optional.of(function));
    var qualifier = mock(Expression.class);
    when(qualifier.typeV2()).thenReturn(owner);
    var name = mock(Name.class);
    when(name.name()).thenReturn("value");
    var expression = mock(QualifiedExpression.class);
    when(expression.qualifier()).thenReturn(qualifier);
    when(expression.name()).thenReturn(name);
    return calculator.calculate(expression);
  }

  @Test
  void calculate_basicMemberAccess_returnsMethodType() {
    FileInput fileInput = parseAndInferTypes("""
      class A:
        def foo(self): ...
      a = A()
      a.foo
      """);

    var qualifiedExpression = getQualifiedExpressionFromStatement(fileInput);
    PythonType result = calculator.calculate(qualifiedExpression);

    assertThat(result).isInstanceOf(FunctionType.class);
    assertThat(result.name()).isEqualTo("foo");
  }

  @Test
  void calculate_propertyDecorator_returnsReturnType() {
    FileInput fileInput = parseAndInferTypes("""
      class A:
        @property
        def foo(self) -> int:
          return 42
      a = A()
      a.foo
      """);

    var qualifiedExpression = getQualifiedExpressionFromStatement(fileInput);
    PythonType result = calculator.calculate(qualifiedExpression);

    assertThat(result.unwrappedType()).isEqualTo(INT_TYPE);
  }

  @Test
  void calculate_propertySubclassDecorator_returnsReturnType() {
    FileInput fileInput = parseAndInferTypes("""
      class subclass_property(property):...

      class A:
        @subclass_property
        def foo(self) -> int:
          return 42
      a = A()
      a.foo
      """);

    var qualifiedExpression = getQualifiedExpressionFromStatement(fileInput);
    PythonType result = calculator.calculate(qualifiedExpression);

    assertThat(result.unwrappedType()).isEqualTo(INT_TYPE);
  }

  @Test
  void calculate_unknownQualifier_returnsUnknown() {
    FileInput fileInput = parseAndInferTypes("""
      unknown.foo
      """);

    var qualifiedExpression = getQualifiedExpressionFromStatement(fileInput);
    PythonType result = calculator.calculate(qualifiedExpression);

    assertThat(result).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void calculate_nonExistentMember_returnsUnknown() {
    FileInput fileInput = parseAndInferTypes("""
      class A:
        pass
      a = A()
      a.non_existent
      """);

    var qualifiedExpression = getQualifiedExpressionFromStatement(fileInput);
    PythonType result = calculator.calculate(qualifiedExpression);

    assertThat(result).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void calculate_nonExistentMemberOnClass_returnsUnresolvedImportType() {
    FileInput fileInput = parseAndInferTypes("""
      class A:
        pass
      A.non_existent
      """);

    var qualifiedExpression = getQualifiedExpressionFromStatement(fileInput);
    PythonType result = calculator.calculate(qualifiedExpression);

    assertThat(result).isInstanceOf(UnknownType.UnresolvedImportType.class);
    var unresolvedType = (UnknownType.UnresolvedImportType) result;
    assertThat(unresolvedType.importPath()).isEqualTo("my_package.mod.A.non_existent");
  }

  @Test
  void calculate_unknownMemberOnKnownModule_returnsUnresolvedImportType() {
    FileInput fileInput = parseAndInferTypes("""
      import os
      os.nonexistent_member
      """);

    var qualifiedExpression = getQualifiedExpressionFromStatement(fileInput);
    PythonType result = calculator.calculate(qualifiedExpression);

    assertThat(result).isInstanceOf(UnknownType.UnresolvedImportType.class);
    var unresolvedType = (UnknownType.UnresolvedImportType) result;
    assertThat(unresolvedType.importPath()).isEqualTo("os.nonexistent_member");
  }

  @Test
  void unionAnnotatedMemberIsResolvedDuringInference() {
    var fileInput = parseAndInferTypes("""
      import ast
      node = ast.ImportFrom(module="os", names=[], level=0)
      node.module()
      """);

    var result = getQualifiedExpressionFromStatement(fileInput).typeV2();

    assertThat(result).isInstanceOfSatisfying(UnionType.class, union -> {
      assertThat(union.candidates()).allSatisfy(candidate -> assertThat(candidate).isInstanceOf(ObjectType.class));
      assertThat(union.candidates()).extracting(candidate -> candidate.unwrappedType().name())
        .containsExactlyInAnyOrder("str", "NoneType");
    });
    assertThat(result.hasMember("__call__")).isEqualTo(TriBool.FALSE);
  }

  @Test
  void aliasAnnotatedMemberPreservesItsUnionValues() {
    var fileInput = parseAndInferTypes("""
      import urllib.request
      request = urllib.request.Request("https://example.com")
      request.data
      """);

    var result = getQualifiedExpressionFromStatement(fileInput).typeV2();

    assertThat(result).isInstanceOfSatisfying(UnionType.class, union -> {
      assertThat(union.candidates()).allSatisfy(candidate -> assertThat(candidate).isInstanceOf(ObjectType.class));
      assertThat(union.candidates()).extracting(candidate -> candidate.unwrappedType().name())
        .contains("bytes", "NoneType");
    });
    assertThat(result.hasMember("__call__")).isEqualTo(TriBool.FALSE);
  }

  private static QualifiedExpression getQualifiedExpressionFromStatement(FileInput fileInput) {
    return PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Tree.Kind.QUALIFIED_EXPR));
  }
}
