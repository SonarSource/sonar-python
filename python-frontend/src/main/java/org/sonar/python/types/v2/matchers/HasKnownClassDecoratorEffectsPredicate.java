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
package org.sonar.python.types.v2.matchers;

import java.util.List;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.SelfType;
import org.sonar.plugins.python.api.types.v2.TypeOrigin;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.UnknownType;

/**
 * Establishes known decorator effects for operand/item checks by stub origin and canonical callable equality.
 * TRUE means all decorators are recognized or class decorators do not apply; FALSE means a decorator
 * has unaccounted-for effects; UNKNOWN means the input type is unresolved.
 * Instances are checked through their class. This does not establish unchanged runtime instance-checking
 * behavior (notably for runtime_checkable).
 */
public class HasKnownClassDecoratorEffectsPredicate implements TypePredicate {
  private static final List<String> CLASS_PRESERVING_DECORATORS = List.of(
    "typing.final", "typing_extensions.final", "typing.disjoint_base", "typing_extensions.disjoint_base",
    "typing.runtime_checkable", "typing_extensions.runtime_checkable");
  private static final List<String> STUB_ONLY_DECORATORS = List.of("typing.type_check_only");

  @Override
  public TriBool check(PythonType type, TypePredicateContext ctx) {
    if (type instanceof ObjectType objectType) {
      return TypePredicateUtils.evaluate(this, objectType.unwrappedType(), ctx);
    }
    if (type instanceof SelfType selfType) {
      return TypePredicateUtils.evaluate(this, selfType.innerType(), ctx);
    }
    if (type instanceof UnknownType || type instanceof UnionType) {
      return TriBool.UNKNOWN;
    }
    return type instanceof ClassType classType ? checkDecorators(classType, ctx) : TriBool.TRUE;
  }

  private static TriBool checkDecorators(ClassType type, TypePredicateContext ctx) {
    if (type.hasDecorators() && type.decorators().isEmpty()) {
      return TriBool.FALSE;
    }
    TypePredicate knownDecorator = (decoratorType, context) -> {
      // Like ClassType's member queries, use the source definition to distinguish user classes from stubs.
      // Typeshed uses type-only protocol bases in runtime class hierarchies, such as numbers.Real.
      if (matchesKnownCallable(decoratorType, context, CLASS_PRESERVING_DECORATORS)
        || (type.definitionLocation().isEmpty() && matchesKnownCallable(decoratorType, context, STUB_ONLY_DECORATORS))) {
        return TriBool.TRUE;
      }
      return TriBool.UNKNOWN;
    };
    for (var decorator : type.decorators()) {
      if (decorator.calleeType().isPresent()
        || !TypePredicateUtils.evaluate(knownDecorator, decorator.expressionType().type(), ctx).isTrue()) {
        return TriBool.FALSE;
      }
    }
    return TriBool.TRUE;
  }

  private static boolean matchesKnownCallable(PythonType type, TypePredicateContext ctx, List<String> names) {
    if (!(type instanceof FunctionType function) || function.typeOrigin() != TypeOrigin.STUB) {
      return false;
    }
    return names.stream().map(ctx.typeTable()::getType).anyMatch(expected ->
      type.equals(expected) || (expected instanceof UnionType union && union.candidates().contains(type)));
  }
}
