/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.python.semantic.v2.types.typecalculator;

import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.SelfType;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.tree.CallExpressionImpl;
import org.sonar.python.types.v2.TypesTestUtils;
import org.sonar.python.types.v2.matchers.TypePredicateContext;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.InstanceOfAssertFactories.type;
import static org.sonar.python.PythonTestUtils.getFirstDescendant;
import static org.sonar.python.PythonTestUtils.getLastDescendant;
import static org.sonar.python.types.v2.TypesTestUtils.INT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.LIST_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE;
import static org.sonar.python.types.v2.TypesTestUtils.parseAndInferTypes;

class CallReturnTypeCalculatorTest {

  private final TypePredicateContext typePredicateContext = TypePredicateContext.of(PROJECT_LEVEL_TYPE_TABLE);

  @Test
  void computeCallExpressionType_classInstantiation() {
    FileInput fileInput = parseAndInferTypes("""
      class MyClass:
        pass
      MyClass()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);
    ClassDef classDef = getFirstDescendant(fileInput, ClassDef.class::isInstance);
    ClassType classType = (ClassType) classDef.name().typeV2();

    PythonType result = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);

    assertThat(result).isInstanceOf(ObjectType.class);
    assertThat(((ObjectType) result).type()).isEqualTo(classType);
  }

  @Test
  void computeCallExpressionType_functionCall() {
    FileInput fileInput = parseAndInferTypes("""
      def foo() -> int: ...
      foo()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);

    PythonType result = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);
    assertThat(result).isInstanceOf(ObjectType.class);
    assertThat(((ObjectType) result).unwrappedType()).isEqualTo(TypesTestUtils.INT_TYPE);
  }

  @Test
  void computeCallExpressionType_unknownCallee() {
    FileInput fileInput = parseAndInferTypes("""
      unknown_func()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);

    PythonType result = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);
    assertThat(result).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void computeCallExpressionType_selfType_collapsedForInstanceMethodCall() {
    FileInput fileInput = parseAndInferTypes("""
      from typing import Self
      class A:
        def foo(self) -> Self: ...
      a = A()
      a.foo()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, tree -> tree instanceof CallExpression ce && ce.callee() instanceof QualifiedExpression);
    ClassDef classDef = getFirstDescendant(fileInput, ClassDef.class::isInstance);
    ClassType classA = (ClassType) classDef.name().typeV2();

    PythonType result = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);

    assertThat(result).isInstanceOf(ObjectType.class);
    ObjectType objectType = (ObjectType) result;
    assertThat(objectType.type()).isEqualTo(classA);
    assertThat(objectType.type()).isNotInstanceOf(SelfType.class);
  }

  @Test
  void computeCallExpressionType_selfType_collapsedWithInheritance() {
    FileInput fileInput = parseAndInferTypes("""
      from typing import Self
      class A:
        def foo(self) -> Self: ...
      class B(A):
        def bar(self): ...
      b = B()
      b.foo()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);
    ClassDef classDefB = getFirstDescendant(fileInput, tree -> tree instanceof ClassDef classDef && classDef.name().name().equals("B"));
    ClassType classB = (ClassType) classDefB.name().typeV2();

    PythonType result = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);

    assertThat(result).isInstanceOf(ObjectType.class);
    ObjectType objectType = (ObjectType) result;
    assertThat(objectType.type()).isEqualTo(classB);
  }

  @Test
  void computeCallExpressionType_selfType_notCollapsedForStaticMethod() {
    FileInput fileInput = parseAndInferTypes("""
      from typing import Self
      class A:
        @staticmethod
        def foo() -> Self: ...
      A.foo()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);

    PythonType result = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);
    assertThat(result).isSameAs(PythonType.UNKNOWN);
  }

  @Test
  void computeCallExpressionType_selfType_receiverIsClassType() {
    FileInput fileInput = parseAndInferTypes("""
      from typing import Self
      class A:
        def foo(self) -> Self: ...
      A.foo()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);

    PythonType result = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);
    assertThat(result)
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isInstanceOf(ClassType.class);
  }

  @Test
  void computeCallExpressionType_selfType_reassignedFunctionYieldsUnknown() {
    FileInput fileInput = parseAndInferTypes("""
      from typing import Self
      class A:
        def foo(self) -> Self: ...
      bar = A.foo
      bar()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);

    PythonType result = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);
    assertThat(result).isSameAs(PythonType.UNKNOWN);
  }

  @Test
  void computeCallExpressionType_selfType_unionWithSelfType() {
    FileInput fileInput = parseAndInferTypes("""
      from typing import Self
      class A:
        def foo(self) -> Self | int: ...

      a = A()
      a.foo()
      """);

    ClassDef classDef = getFirstDescendant(fileInput, ClassDef.class::isInstance);
    ClassType classType = (ClassType) classDef.name().typeV2();

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);

    PythonType result = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);

    Set<PythonType> candidates = assertThat(result)
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .asInstanceOf(type(UnionType.class))
      .extracting(UnionType::candidates)
      .actual();

    assertThat(candidates)
      .satisfiesOnlyOnce(type -> assertThat(type).is(TypesTestUtils.objectTypeOf(classType)))
      .satisfiesOnlyOnce(type -> assertThat(type).is(TypesTestUtils.objectTypeOf(INT_TYPE)))
      .hasSize(2);
  }

  @Test
  void computeCallExpressionType_selfType_unionWithSelfTypeInList() {
    FileInput fileInput = parseAndInferTypes("""
      from typing import Self
      class A:
        def foo(self) -> list[Self] | int: ...

      a = A()
      a.foo()
      """);

    ClassDef classDef = getFirstDescendant(fileInput, ClassDef.class::isInstance);
    ClassType classType = (ClassType) classDef.name().typeV2();

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);

    PythonType result = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);

    Set<PythonType> candidates = assertThat(result)
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .asInstanceOf(type(UnionType.class))
      .extracting(UnionType::candidates)
      .actual();
    assertThat(candidates)
      .satisfiesOnlyOnce(type -> assertThat(type).is(TypesTestUtils.objectTypeOf(INT_TYPE)))
      .satisfiesOnlyOnce(type -> assertThat(type)
        .is(TypesTestUtils.objectTypeOf(LIST_TYPE))
        .asInstanceOf(type(ObjectType.class))
        .extracting(ObjectType::attributes)
        .isEqualTo(List.of(ObjectType.fromType(classType))))
      .hasSize(2);
  }

  @Test
  void computeCallExpressionType_selfType_listWithSelfType() {
    FileInput fileInput = parseAndInferTypes("""
      from typing import Self
      class A:
        def foo(self) -> list[Self]: ...

      a = A()
      a.foo()
      """);

    ClassDef classDef = getFirstDescendant(fileInput, ClassDef.class::isInstance);
    ClassType classType = (ClassType) classDef.name().typeV2();

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);

    PythonType result = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);

    assertThat(result)
      .asInstanceOf(type(ObjectType.class))
      .is(TypesTestUtils.objectTypeOf(LIST_TYPE))
      .extracting(ObjectType::attributes)
      .isEqualTo(List.of(ObjectType.fromType(classType)));
  }

  @Test
  void computeCallExpressionType_selfTypeInGenerator_typeshedFunction() {
    PythonType generatorType = PROJECT_LEVEL_TYPE_TABLE.getType("typing.Generator");
    assertThat(generatorType).isNotEqualTo(PythonType.UNKNOWN);

    PythonType pathType = PROJECT_LEVEL_TYPE_TABLE.getType("pathlib.Path");
    assertThat(pathType).isNotEqualTo(PythonType.UNKNOWN);

    FileInput fileInput = parseAndInferTypes("""
      import pathlib
      path = pathlib.Path(".")
      globGenerator = path.glob("*.py")
      """);

    Name globGeneratorName = PythonTestUtils.getLastDescendant(fileInput, tree -> tree instanceof Name name && "globGenerator".equals(name.name()));
    PythonType globGeneratorType = globGeneratorName.typeV2();

    assertThat(globGeneratorType).is(TypesTestUtils.objectTypeOf(generatorType));
    ObjectType objectType = (ObjectType) globGeneratorType;
    assertThat(objectType.attributes())
      .element(0)
      .isEqualTo(pathType);
  }

  @Test
  void computeCallExpressionType_selfType_typeshedFunction() {
    PythonType pathType = PROJECT_LEVEL_TYPE_TABLE.getType("pathlib.Path");
    assertThat(pathType).isNotEqualTo(PythonType.UNKNOWN);

    FileInput fileInput = parseAndInferTypes("""
      import pathlib
      path = pathlib.Path(".")
      renamedSelf = path.rename(...)
      """);

    Name renamedSelfName = PythonTestUtils.getLastDescendant(fileInput, tree -> tree instanceof Name name && "renamedSelf".equals(name.name()));
    PythonType renamedSelfType = renamedSelfName.typeV2();

    assertThat(renamedSelfType).is(TypesTestUtils.objectTypeOf(pathType));
    ObjectType objectType = (ObjectType) renamedSelfType;
    assertThat(objectType.attributes()).isEmpty();
  }

  @Test
  void computeCallExpressionType_selfTypeInOverloadedFunction_typeshedFunction() {
    PythonType moduleType = PROJECT_LEVEL_TYPE_TABLE.getType("torch.nn.modules.Module");
    assertThat(moduleType).isNotEqualTo(PythonType.UNKNOWN);

    FileInput fileInput = parseAndInferTypes("""
      from torch.nn.modules import Module
      module = Module()
      toModule = module.to(...)
      """);

    Name toModuleName = PythonTestUtils.getLastDescendant(fileInput, tree -> tree instanceof Name name && "toModule".equals(name.name()));
    PythonType toModuleType = toModuleName.typeV2();

    assertThat(toModuleType).is(TypesTestUtils.objectTypeOf(moduleType));
  }

  @Test
  void computeCallExpressionType_methodCall() {
    FileInput fileInput = parseAndInferTypes("""
      class MyClass:
        def method(self) -> str: ...
      obj = MyClass()
      obj.method()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);

    PythonType result = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);
    assertThat(result).isInstanceOf(ObjectType.class);
    assertThat(((ObjectType) result).unwrappedType()).isEqualTo(TypesTestUtils.STR_TYPE);
  }

  @Test
  void computeCallExpressionType_unionType_returnsUnionOfReturnTypes() {
    FileInput fileInput = parseAndInferTypes("""
      class A: ...
      class B: ...
      if cond:
        x = A
      else:
        x = B
      x()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);
    ClassDef classDefA = getFirstDescendant(fileInput, tree -> tree instanceof ClassDef cd && cd.name().name().equals("A"));
    ClassDef classDefB = getFirstDescendant(fileInput, tree -> tree instanceof ClassDef cd && cd.name().name().equals("B"));
    ClassType classA = (ClassType) classDefA.name().typeV2();
    ClassType classB = (ClassType) classDefB.name().typeV2();

    PythonType result = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);

    assertThat(result).isInstanceOf(UnionType.class);
    UnionType resultUnion = (UnionType) result;
    assertThat(resultUnion.candidates())
      .extracting(PythonType::unwrappedType)
      .containsExactlyInAnyOrder(classA, classB);
  }

  @Test
  void computeCallExpressionType_objectTypeWithCallable_resolvesCallMember() {
    FileInput fileInput = parseAndInferTypes("""
      class MyCallable:
        def __call__(self) -> int: ...
      obj = MyCallable()
      obj()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, tree -> tree instanceof CallExpression ce
      && ce.callee().firstToken().value().equals("obj"));

    PythonType result = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);
    assertThat(result).isInstanceOf(ObjectType.class);
    assertThat(result.unwrappedType()).isEqualTo(TypesTestUtils.INT_TYPE);
  }

  @Test
  void computeCallExpressionType_selfType_collapsedForClassMethod() {
    FileInput fileInput = parseAndInferTypes("""
      from typing import Self
      class A:
        @classmethod
        def foo(cls) -> Self: return cls()
      A.foo()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);
    ClassDef classDef = getFirstDescendant(fileInput, ClassDef.class::isInstance);
    ClassType classA = (ClassType) classDef.name().typeV2();

    PythonType result = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);

    assertThat(result).isInstanceOf(ObjectType.class);
    ObjectType objectType = (ObjectType) result;
    assertThat(objectType.type()).isEqualTo(classA);
    assertThat(objectType.type()).isNotInstanceOf(SelfType.class);
  }

  @Test
  void computeCallExpressionType_selfType_collapsedForClassMethodWithInheritance() {
    FileInput fileInput = parseAndInferTypes("""
      from typing import Self
      class A:
        @classmethod
        def foo(cls) -> Self: return cls()
      class B(A): ...
      B.foo()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);
    ClassDef classDefB = getFirstDescendant(fileInput, tree -> tree instanceof ClassDef classDef && classDef.name().name().equals("B"));
    ClassType classB = (ClassType) classDefB.name().typeV2();

    PythonType result = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);

    assertThat(result).isInstanceOf(ObjectType.class);
    ObjectType objectType = (ObjectType) result;
    assertThat(objectType.type()).isEqualTo(classB);
  }

  @Test
  void computeCallExpressionType_selfType_classMethodCalledOnInstance() {
    FileInput fileInput = parseAndInferTypes("""
      from typing import Self
      class A:
        @classmethod
        def foo(cls) -> Self: return cls()
      a = A()
      a.foo()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);
    ClassDef classDef = getFirstDescendant(fileInput, ClassDef.class::isInstance);
    ClassType classA = (ClassType) classDef.name().typeV2();

    PythonType result = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);

    assertThat(result).isInstanceOf(ObjectType.class);
    ObjectType objectType = (ObjectType) result;
    assertThat(objectType.type()).isEqualTo(classA);
  }

  @Test
  void callNonCallableExpression() {
    FileInput fileInput = parseAndInferTypes("""
      from typing import Self
      class MyClass:
        def __call__(self) -> Self: ...
      obj = MyClass()
      obj()
      """);
    ClassDef classDef = getFirstDescendant(fileInput, ClassDef.class::isInstance);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);
    PythonType result = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);
    assertThat(result)
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isEqualTo(classDef.name().typeV2());
  }

  @Test
  void computeCallExpressionType_revealType_transparent() {
    FileInput fileInput = parseAndInferTypes("""
      x = 42
      reveal_type(x)
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);

    PythonType result = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);
    assertThat(result).isInstanceOf(ObjectType.class);
    assertThat(((ObjectType) result).unwrappedType()).isEqualTo(TypesTestUtils.INT_TYPE);
  }

  @Test
  void computeCallExpressionType_revealType_inTypeAnnotation() {
    FileInput fileInput = parseAndInferTypes("""
      def foo(x: reveal_type(int)): ...
      """);

    CallExpressionImpl callExpr = getFirstDescendant(fileInput, CallExpression.class::isInstance);

    PythonType result = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);
    assertThat(result).isEqualTo(TypesTestUtils.INT_TYPE);
  }

  @Test
  void computeCallExpressionType_revealType_noArguments() {
    FileInput fileInput = parseAndInferTypes("""
      reveal_type()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);

    PythonType result = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);
    assertThat(result).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void computeCallExpressionType_revealType_locallyDefined_notTransparent() {
    FileInput fileInput = parseAndInferTypes("""
      def reveal_type(x) -> str:
        return str(type(x))
      x = 42
      reveal_type(x)
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);

    PythonType result = CallReturnTypeCalculator.computeCallExpressionType(callExpr, typePredicateContext);
    assertThat(result).isInstanceOf(ObjectType.class);
    assertThat(((ObjectType) result).unwrappedType()).isEqualTo(TypesTestUtils.STR_TYPE);
  }
}
