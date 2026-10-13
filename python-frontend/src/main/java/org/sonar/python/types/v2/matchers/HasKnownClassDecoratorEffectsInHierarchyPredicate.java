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

import java.util.ArrayDeque;
import java.util.HashSet;
import java.util.Set;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.SelfType;
import org.sonar.plugins.python.api.types.v2.UnionType;

/** Checks all possible bases before establishing known inherited operand/item decorator effects. */
public class HasKnownClassDecoratorEffectsInHierarchyPredicate implements TypePredicate {
  private final HasKnownClassDecoratorEffectsPredicate direct = new HasKnownClassDecoratorEffectsPredicate();

  @Override
  public TriBool check(PythonType type, TypePredicateContext ctx) {
    if (type instanceof ObjectType objectType) {
      return TypePredicateUtils.evaluate(this, objectType.unwrappedType(), ctx);
    }
    if (type instanceof SelfType selfType) {
      return TypePredicateUtils.evaluate(this, selfType.innerType(), ctx);
    }
    TriBool result = direct.check(type, ctx);
    if (!result.isTrue() || !(type instanceof ClassType classType)) {
      return result;
    }
    Set<PythonType> visited = new HashSet<>();
    visited.add(type);
    var pending = new ArrayDeque<PythonType>();
    classType.superClasses().forEach(base -> pending.add(base.type()));
    while (!pending.isEmpty()) {
      PythonType current = pending.removeFirst();
      if (!visited.add(current)) {
        continue;
      }
      switch (current) {
        case UnionType union -> {
          // Unlike the outer expression union, every possible base contributes inherited effects.
          pending.addAll(union.candidates());
        }
        case ClassType base -> {
          result = result.logicalAnd(direct.check(base, ctx));
          if (result.isFalse()) {
            return TriBool.FALSE;
          }
          base.superClasses().forEach(parent -> pending.add(parent.type()));
        }
        default -> result = TriBool.UNKNOWN;
      }
    }
    return result;
  }
}
