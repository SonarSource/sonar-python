/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.checks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.tree.YieldExpression;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.types.InferredTypes;

import static org.sonar.python.types.InferredTypes.containsDeclaredType;

@Rule(key = "S5886")
public class FunctionReturnTypeCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Return a value of type \"%s\" instead of \"%s\" or update function \"%s\" type hint.";
  private static final List<String> ITERABLE_TYPES = Arrays.asList("typing.Generator", "typing.Iterator", "typing.Iterable");
  private static final List<String> ASYNC_ITERABLE_TYPES = Arrays.asList("typing.AsyncGenerator", "typing.AsyncIterator", "typing.AsyncIterable");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      Symbol symbol = functionDef.name().symbol();
      if (symbol == null || !symbol.is(Symbol.Kind.FUNCTION)) {
        return;
      }
      FunctionSymbolImpl functionSymbol = (FunctionSymbolImpl) symbol;
      InferredType declaredReturnType = functionSymbol.declaredReturnType();
      if (declaredReturnType == InferredTypes.anyType()) {
        return;
      }
      ReturnTypeVisitor returnTypeVisitor = new ReturnTypeVisitor(declaredReturnType);
      functionDef.body().accept(returnTypeVisitor);
      raiseIssues(ctx, functionDef, declaredReturnType, returnTypeVisitor);
    });
  }

  private static void raiseIssues(SubscriptionContext ctx, FunctionDef functionDef, InferredType declaredReturnType, ReturnTypeVisitor returnTypeVisitor) {
    String functionName = functionDef.name().name();
    String returnTypeName = InferredTypes.typeName(declaredReturnType);
    if (!returnTypeVisitor.yieldExpressions.isEmpty()) {
      boolean isAsyncFunction = functionDef.asyncKeyword() != null;
      String recommendedSuperType = isAsyncFunction ? "typing.AsyncGenerator" : "typing.Generator";
      // Here we should probably use an equivalent of "canBeOrExtend" (accepting uncertainty) instead of "mustBeOrExtend"
      if (ITERABLE_TYPES.stream().anyMatch(declaredReturnType::mustBeOrExtend) || ASYNC_ITERABLE_TYPES.stream().anyMatch(declaredReturnType::mustBeOrExtend)) {
        if (isMixedUpAnnotation(isAsyncFunction, declaredReturnType)) {
          returnTypeVisitor.yieldExpressions
            .forEach(y -> {
              PreciseIssue issue =
                ctx.addIssue(y, String.format("Annotate function \"%s\" with \"%s\" or one of its supertypes.", functionName, recommendedSuperType));
              addSecondaries(issue, functionDef);
            });
        }
        return;
      }
      returnTypeVisitor.yieldExpressions
        .forEach(y -> {
          PreciseIssue issue =
            ctx.addIssue(y, String.format("Remove this yield statement or annotate function \"%s\" with \"%s\" or one of its supertypes.", functionName, recommendedSuperType));
          addSecondaries(issue, functionDef);
        });
    }
    returnTypeVisitor.invalidReturnStatements.forEach(i -> {
      PreciseIssue issue;
      if (i.expressions().size() > 1) {
        issue = ctx.addIssue(i, String.format(MESSAGE, returnTypeName, "tuple", functionName));
      } else if (i.expressions().size() == 1 && InferredTypes.typeName(i.expressions().get(0).type()) != null) {
        issue = ctx.addIssue(i.expressions().get(0), String.format(MESSAGE, returnTypeName, InferredTypes.typeName(i.expressions().get(0).type()), functionName));
      } else {
        issue = ctx.addIssue(i, String.format("Return a value of type \"%s\" or update function \"%s\" type hint.", returnTypeName, functionName));
      }
      addSecondaries(issue, functionDef);
    });
  }

  private static boolean isMixedUpAnnotation(boolean isAsyncFunction, InferredType declaredReturnType) {
    return isAsyncFunction ? ITERABLE_TYPES.stream().anyMatch(declaredReturnType::mustBeOrExtend) : ASYNC_ITERABLE_TYPES.stream().anyMatch(declaredReturnType::mustBeOrExtend);
  }

  private static void addSecondaries(PreciseIssue issue, FunctionDef functionDef) {
    issue.secondary(functionDef.name(), "Function definition.");
    TypeAnnotation returnTypeAnnotation = functionDef.returnTypeAnnotation();
    if (returnTypeAnnotation != null) {
      issue.secondary(returnTypeAnnotation.expression(), "Type hint.");
    }
  }

  private static class ReturnTypeVisitor extends BaseTreeVisitor {

    InferredType returnType;
    List<YieldExpression> yieldExpressions = new ArrayList<>();
    List<ReturnStatement> invalidReturnStatements = new ArrayList<>();

    ReturnTypeVisitor(InferredType returnType) {
      this.returnType = returnType;
    }

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      // Don't visit nested functions
    }

    @Override
    public void visitReturnStatement(ReturnStatement returnStatement) {
      List<Expression> expressions = returnStatement.expressions();
      if (expressions.isEmpty()) {
        if (!InferredTypes.NONE.isCompatibleWith(returnType)) {
          invalidReturnStatements.add(returnStatement);
        }
      } else if (!returnStatement.commas().isEmpty()) {
        // Hardcoded "tuple" type due to a limitation on extracting type information from tuple literals
        if (!InferredTypes.TUPLE.isCompatibleWith(returnType)) {
          invalidReturnStatements.add(returnStatement);
        }
      } else {
        Expression expression = expressions.get(0);
        InferredType inferredType = expression.type();
        if (returnType.mustBeOrExtend("typing.TypedDict")) {
          // Avoid FPs for TypedDict
          return;
        }

        if (!containsDeclaredType(inferredType) && !inferredType.isCompatibleWith(returnType)) {
          invalidReturnStatements.add(returnStatement);
        }
      }
      super.visitReturnStatement(returnStatement);
    }

    @Override
    public void visitYieldExpression(YieldExpression yieldExpression) {
      yieldExpressions.add(yieldExpression);
      super.visitYieldExpression(yieldExpression);
    }
  }
}
