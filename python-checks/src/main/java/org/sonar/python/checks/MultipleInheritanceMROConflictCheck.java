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

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;

@Rule(key = "S8511")
public class MultipleInheritanceMROConflictCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Reorder or remove base classes to fix this MRO conflict.";
  private static final String SECONDARY_MESSAGE = "This base class is an ancestor of another listed base class appearing after it.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, MultipleInheritanceMROConflictCheck::checkClassDef);
  }

  private static void checkClassDef(SubscriptionContext ctx) {
    ClassDef classDef = (ClassDef) ctx.syntaxNode();
    ArgList argList = classDef.args();
    if (argList == null) {
      return;
    }

    List<Expression> bases = collectPositionalBases(argList);
    if (bases.size() < 2) {
      return;
    }

    List<ClassType> types = new ArrayList<>(
            bases.stream().map(MultipleInheritanceMROConflictCheck::resolveClassType).toList()
    );

    int conflictIndex = findAncestorConflictIndex(types);
    if (hasMroConflict(types, conflictIndex)) {
      PreciseIssue issue = ctx.addIssue(classDef.name(), MESSAGE);
      if (conflictIndex >= 0) {
        issue.secondary(bases.get(conflictIndex), SECONDARY_MESSAGE);
      }
    }
  }

  private static List<Expression> collectPositionalBases(ArgList argList) {
    List<Expression> bases = new ArrayList<>();
    argList.arguments().stream()
            .filter(argument ->
                    argument instanceof RegularArgument regularArgument
                            && regularArgument.keywordArgument() == null
            )
            .map(argument -> ((RegularArgument) argument).expression())
            .forEach(bases::add);
    return bases;
  }

  /**
   * Detects an MRO conflict: either via C3 when every base is fully resolved, or otherwise only
   * when {@link #findAncestorConflictIndex} finds an earlier base that is a strict superclass of a
   * later base. Both paths use a runtime-faithful view of built-in containers — see
   * {@link ClassType#wouldHaveValidMro(List)}.
   */
  private static boolean hasMroConflict(List<ClassType> types, int conflictIndex) {
    if (isFullyResolved(types)) {
      return !ClassType.wouldHaveValidMro(types);
    }
    return conflictIndex >= 0;
  }

  /**
   * Returns {@code true} if every base class is fully resolved (non-null, with a fully known type
   * hierarchy). Only when this is true can we run the complete C3 algorithm safely.
   */
  private static boolean isFullyResolved(List<ClassType> types) {
    for (ClassType type : types) {
      if (type == null || type.hasUnresolvedHierarchy()) {
        return false;
      }
    }
    return true;
  }

  /**
   * Returns the index {@code i} of the first base class that is an ancestor of some later base
   * class at index {@code j > i}, or {@code -1} if no such pair exists. Paths from any later base
   * that pass through a virtual-ABC-subclassing builtin are cut off, so typeshed-only ABC edges
   * (e.g. {@code dict → MutableMapping}) are not treated as real ancestry — see
   * {@link #isOrExtendsClassAtRuntime(ClassType, ClassType)}.
   */
  private static int findAncestorConflictIndex(List<ClassType> types) {
    for (int i = 0; i < types.size() - 1; i++) {
      if (laterBaseExtendsEarlier(types, i)) {
        return i;
      }
    }
    return -1;
  }

  private static boolean laterBaseExtendsEarlier(List<ClassType> types, int i) {
    ClassType typeI = types.get(i);
    if (typeI == null) {
      return false;
    }
    for (int j = i + 1; j < types.size(); j++) {
      ClassType typeJ = types.get(j);
      if (typeJ != null && isOrExtendsClassAtRuntime(typeJ, typeI)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns {@code true} if {@code ancestor} appears in {@code candidate}'s runtime type hierarchy.
   * Traversal stops at virtual-ABC-subclassing builtins so their typeshed-only ABC parents are not
   * reachable; see {@link ClassType#wouldHaveValidMro(List)}. Uses reference equality
   * since {@link ClassType} instances are canonical within a single analysis.
   */
  private static boolean isOrExtendsClassAtRuntime(ClassType candidate, ClassType ancestor) {
    Set<PythonType> visited = new HashSet<>();
    Deque<PythonType> queue = new ArrayDeque<>();
    queue.add(candidate);
    while (!queue.isEmpty()) {
      PythonType current = queue.poll();
      if (!visited.add(current)) {
        continue;
      }
      if (current == ancestor) {
        return true;
      }
      if (current instanceof ClassType ct && !ct.isVirtualAbcSubclassingBuiltin()) {
        enqueueDirectSuperclasses(ct, queue);
      }
    }
    return false;
  }

  private static void enqueueDirectSuperclasses(ClassType ct, Deque<PythonType> queue) {
    for (TypeWrapper sw : ct.superClasses()) {
      queue.add(sw.type());
    }
  }

  @CheckForNull
  private static ClassType resolveClassType(Expression expression) {
    PythonType type = expression.typeV2();
    return (type instanceof ClassType classType) ? classType : null;
  }
}
