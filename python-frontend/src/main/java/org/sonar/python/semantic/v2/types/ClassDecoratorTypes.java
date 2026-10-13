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
package org.sonar.python.semantic.v2.types;

import java.util.List;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.DecoratorInfo;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;

/** Completes source decorator types after inference in their enclosing evaluation scope. */
public final class ClassDecoratorTypes {
  private ClassDecoratorTypes() {}

  public static List<DecoratorInfo> deferred(ClassDef definition) {
    return definition.decorators().stream().map(decorator -> new DecoratorInfo(
      new DeferredDecoratorTypeWrapper(),
      unwrapParentheses(decorator.expression()) instanceof CallExpression ? new DeferredDecoratorTypeWrapper() : null)).toList();
  }

  public static void completeScope(Tree scope) {
    scope.accept(new BaseTreeVisitor() {
      @Override
      public void visitFunctionDef(FunctionDef definition) {
        // Nested function bodies are inferred separately. Their decorators belong to this scope,
        // but this visitor only completes class decorator metadata.
      }

      @Override
      public void visitClassDef(ClassDef definition) {
        if (definition.name().typeV2() instanceof ClassType type) {
          for (int i = 0; i < type.decorators().size(); i++) {
            var expression = unwrapParentheses(definition.decorators().get(i).expression());
            var info = type.decorators().get(i);
            if (info.expressionType() instanceof DeferredDecoratorTypeWrapper deferred) {
              deferred.complete(expression.typeV2());
            }
            if (expression instanceof CallExpression call) {
              info.calleeType().filter(DeferredDecoratorTypeWrapper.class::isInstance).map(DeferredDecoratorTypeWrapper.class::cast)
                .ifPresent(deferred -> deferred.complete(call.callee().typeV2()));
            }
          }
        }
        super.visitClassDef(definition);
      }
    });
  }

  private static Expression unwrapParentheses(Expression expression) {
    while (expression instanceof ParenthesizedExpression parenthesized) {
      expression = parenthesized.expression();
    }
    return expression;
  }

  private static final class DeferredDecoratorTypeWrapper implements TypeWrapper {
    private TypeWrapper inferredType = UNKNOWN_TYPE_WRAPPER;
    private boolean completed;

    private void complete(PythonType type) {
      if (completed) {
        throw new IllegalStateException("Decorator type already completed");
      }
      inferredType = TypeWrapper.of(type);
      completed = true;
    }

    @Override
    public PythonType type() {
      // An early read must not freeze the slot to UNKNOWN.
      return inferredType.type();
    }
  }
}
