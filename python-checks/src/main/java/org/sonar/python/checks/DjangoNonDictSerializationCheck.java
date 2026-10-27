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
import java.util.Objects;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentExpression;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.types.v2.matchers.InternalTypeMatchers;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6560")
public class DjangoNonDictSerializationCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use a dictionary object here, or set the \"safe\" flag to False.";
  private static final TypeMatcher JSON_RESPONSE_TYPE_MATCHER = TypeMatchers.isType("django.http.JsonResponse");
  private static final TypeMatcher DICTIONARY_TYPE_MATCHER = TypeMatchers.isObjectInstanceOf("dict");
  private static final TypeMatcher UNION_CONTAINING_DICTIONARY_TYPE_MATCHER = InternalTypeMatchers.isAnyTypeInUnionSatisfying(DICTIONARY_TYPE_MATCHER);

  private static final int MAX_RECURSION = 5;

  @Override
  public void initialize(Context context) {

    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      if (JSON_RESPONSE_TYPE_MATCHER.isTrueFor(callExpression.callee(), ctx)) {
        checkForDictSerialization(ctx, callExpression);
      }
    });
  }

  private static void checkForDictSerialization(SubscriptionContext ctx, CallExpression callExpression) {
    RegularArgument safe = TreeUtils.nthArgumentOrKeyword(2, "safe", callExpression.arguments());
    if (safe == null || (safe.expression().is(Tree.Kind.NAME) && "True".equals(((Name) safe.expression()).name()))) {
      RegularArgument dataArg = TreeUtils.nthArgumentOrKeyword(0, "data", callExpression.arguments());
      if (dataArg != null && !couldExpressionBeADict(dataArg.expression(), ctx)) {
        ctx.addIssue(dataArg, MESSAGE);
      }
    }
  }

  private static boolean couldExpressionBeADict(Expression expression, SubscriptionContext ctx) {
    if (expression.is(Tree.Kind.NAME)) {
      return couldDictBeAssignedToDataArg((Name) expression, ctx, 0);
    }
    return couldTypeBeADict(expression, ctx);
  }

  private static boolean couldDictBeAssignedToDataArg(Name dataArg, SubscriptionContext ctx, int recursiveCount) {
    SymbolV2 dataArgSymbol = dataArg.symbolV2();
    if (recursiveCount <= MAX_RECURSION && dataArgSymbol != null) {
      List<Tree> assignmentStmts = dataArgSymbol.usages().stream()
        .filter(usage -> usage.kind() == UsageV2.Kind.ASSIGNMENT_LHS)
        .map(UsageV2::tree)
        .map(usage -> TreeUtils.firstAncestorOfKind(usage, Tree.Kind.ASSIGNMENT_STMT, Tree.Kind.ANNOTATED_ASSIGNMENT, Tree.Kind.ASSIGNMENT_EXPRESSION))
        .filter(Objects::nonNull)
        .toList();
      if (assignmentStmts.size() == 1) {
        Tree assignment = assignmentStmts.get(0);
        Expression assignedValue = getAssignedValue(assignment);
        // We do not yet support checking the types from call expressions
        if (assignedValue != null) {
          if (assignedValue.is(Tree.Kind.NAME)) {
            return couldDictBeAssignedToDataArg((Name) assignedValue, ctx, recursiveCount + 1);
          } else {
            return couldTypeBeADict(assignedValue, ctx);
          }
        }
      }
    }
    return true;
  }

  private static boolean couldTypeBeADict(Expression expression, SubscriptionContext ctx) {
    return expression.is(Tree.Kind.DICTIONARY_LITERAL) || expression.is(Tree.Kind.DICT_COMPREHENSION)
      || DICTIONARY_TYPE_MATCHER.evaluateFor(expression, ctx) != TriBool.FALSE
      || UNION_CONTAINING_DICTIONARY_TYPE_MATCHER.evaluateFor(expression, ctx) != TriBool.FALSE;
  }

  @CheckForNull
  private static Expression getAssignedValue(Tree assignment) {
    if (assignment.is(Tree.Kind.ASSIGNMENT_STMT)) {
      return ((AssignmentStatement) assignment).assignedValue();
    } else if (assignment.is(Tree.Kind.ANNOTATED_ASSIGNMENT)) {
      return ((AnnotatedAssignment) assignment).assignedValue();
    } else {
      return ((AssignmentExpression) assignment).expression();
    }
  }
}
