/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.checks.tests;

import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.tests.UnittestUtils;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.InferredTypes;

import static org.sonar.plugins.python.api.types.BuiltinTypes.NONE_TYPE;

@Rule(key = "S5845")
public class AssertOnDissimilarTypesCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Change this assertion to not compare dissimilar types";
  private static final String MESSAGE_SECONDARY = "Last assignment of \"%s\"";
  private static final Set<String> assertToCheckEquality = Set.of("assertEqual", "assertNotEqual");
  private static final Set<String> assertToCheckIdentity = Set.of("assertIs", "assertIsNot");
  private static final String FIRST_ARG_KEYWORD = "first";
  private static final String SECOND_ARG_KEYWORD = "second";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      if (!callExpression.callee().is(Tree.Kind.QUALIFIED_EXPR)) {
        return;
      }

      QualifiedExpression qualifiedExpression = (QualifiedExpression) callExpression.callee();
      if ((!(qualifiedExpression.qualifier().is(Tree.Kind.NAME) && ((Name) qualifiedExpression.qualifier()).name().equals("self")))
       || !UnittestUtils.isWithinUnittestTestCase(qualifiedExpression)) {
        return;
      }

      checkArguments(ctx, callExpression, qualifiedExpression);
    });
  }

  private static void checkArguments(SubscriptionContext ctx, CallExpression callExpression, QualifiedExpression qualifiedExpression) {
    boolean isAnAssertIdentity = assertToCheckIdentity.contains(qualifiedExpression.name().name());
    boolean isAnAssertEquality = assertToCheckEquality.contains(qualifiedExpression.name().name());

    if (!isAnAssertEquality && !isAnAssertIdentity) {
      return;
    }

    ArgList args = callExpression.argumentList();
    if (args == null) {
      return;
    }

    List<Argument> arguments = args.arguments();
    RegularArgument firstArg = TreeUtils.nthArgumentOrKeyword(0, FIRST_ARG_KEYWORD, arguments);
    RegularArgument secondArg = TreeUtils.nthArgumentOrKeyword(1, SECOND_ARG_KEYWORD, arguments);
    if (firstArg == null || secondArg == null) {
      return;
    }

    Expression left = firstArg.expression();
    Expression right = secondArg.expression();

    if (canArgumentsBeIdentical(left, right)) {
      return;
    }

    if (isAnAssertIdentity || !canArgumentsBeEqual(left, right)) {
      PreciseIssue issue = ctx.addIssue(args, message(left, right));
      getLastAssignment(left).ifPresent(assign -> issue.secondary(assign, String.format(MESSAGE_SECONDARY, ((Name)left).name())));
      getLastAssignment(right).ifPresent(assign -> issue.secondary(assign, String.format(MESSAGE_SECONDARY, ((Name)right).name())));
    }
  }

  private static String message(Expression left, Expression right) {
    String leftTypeName = InferredTypes.typeName(left.type());
    String rightTypeName = InferredTypes.typeName(right.type());
    String message = MESSAGE;
    if (leftTypeName != null && rightTypeName != null) {
      message += " (" + leftTypeName + " and " + rightTypeName + ")";
    }
    message += ".";
    return message;
  }

  private static Optional<AssignmentStatement> getLastAssignment(Expression expr) {
    if (!expr.is(Tree.Kind.NAME)) {
      return Optional.empty();
    }

    Symbol symbol = ((Name) expr).symbol();
    if (symbol == null) {
      return Optional.empty();
    }

    Usage lastAssignment = symbol.usages().stream()
      .sorted(Comparator.comparingInt(u -> u.tree().firstToken().line()))
      .takeWhile(usage -> usage != ((Name) expr).usage())
      .filter(usage -> usage.kind() == Usage.Kind.ASSIGNMENT_LHS)
      .reduce((first, second) -> second)
      .orElse(null);

    return Optional.ofNullable(lastAssignment)
      .map(assignment -> (AssignmentStatement) TreeUtils.firstAncestorOfKind(assignment.tree(), Tree.Kind.ASSIGNMENT_STMT));
  }

  private static boolean canArgumentsBeIdentical(Expression left, Expression right) {
    return left.type().isIdentityComparableWith(right.type()) || left.type().canOnlyBe(NONE_TYPE) || right.type().canOnlyBe(NONE_TYPE);
  }

  private static boolean canArgumentsBeEqual(Expression left, Expression right) {
    String leftCategory = InferredTypes.getBuiltinCategory(left.type());
    String rightCategory = InferredTypes.getBuiltinCategory(right.type());
    boolean leftCanImplementEqOrNe = canImplementEqOrNe(left);
    boolean rightCanImplementEqOrNe = canImplementEqOrNe(right);

    if ((leftCategory != null && leftCategory.equals(rightCategory))) {
      return true;
    }

    return (leftCanImplementEqOrNe || rightCanImplementEqOrNe)
      && (leftCategory == null || rightCategory == null)
      && (leftCategory == null || rightCanImplementEqOrNe)
      && (rightCategory == null || leftCanImplementEqOrNe);
  }

  private static boolean canImplementEqOrNe(Expression expression) {
    InferredType type = expression.type();
    return type.canHaveMember("__eq__") || type.canHaveMember("__ne__");
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }
}
