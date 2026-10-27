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
package org.sonar.python.checks.tests;

import java.util.ArrayDeque;
import java.util.Comparator;
import java.util.Deque;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.CompoundAssignmentStatement;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.WithItem;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.checks.utils.UnittestUtils;
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
      if (!(qualifiedExpression.qualifier() instanceof Name qualifier && "self".equals(qualifier.name()))
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

    if (hasWrittenAttribute(left) || hasWrittenAttribute(right) || canArgumentsBeIdentical(left, right)) {
      return;
    }

    if (isAnAssertIdentity || !canArgumentsBeEqual(left, right)) {
      PreciseIssue issue = ctx.addIssue(args, message(left, right));
      getLastAssignment(left).ifPresent(assign -> issue.secondary(assign, String.format(MESSAGE_SECONDARY, ((Name)left).name())));
      getLastAssignment(right).ifPresent(assign -> issue.secondary(assign, String.format(MESSAGE_SECONDARY, ((Name)right).name())));
    }
  }

  private static boolean hasWrittenAttribute(Expression expression) {
    Deque<String> memberPath = new ArrayDeque<>();
    Expression receiver = Expressions.removeParentheses(expression);
    while (receiver instanceof QualifiedExpression member) {
      memberPath.addFirst(member.name().name());
      receiver = Expressions.removeParentheses(member.qualifier());
    }
    if (memberPath.isEmpty() || !(receiver instanceof Name name) || name.symbolV2() == null) {
      return false;
    }
    // Member types still describe the class declaration after instance writes.
    // Do not infer incompatibility from that stale type. Include writes later in
    // the source, since a loop or closure may execute them before this read.
    return name.symbolV2().usages().stream().anyMatch(usage -> writesMemberPath(usage.tree(), memberPath));
  }

  private static boolean writesMemberPath(Tree receiver, Deque<String> memberPath) {
    for (String memberName : memberPath) {
      while (receiver.parent().is(Tree.Kind.PARENTHESIZED)) {
        receiver = receiver.parent();
      }
      if (!(receiver.parent() instanceof QualifiedExpression member)
        || member.qualifier() != receiver || !memberName.equals(member.name().name())) {
        return false;
      }
      // Replacing an intermediate receiver also invalidates its members' types.
      if (isAssignmentTarget(member)) {
        return true;
      }
      receiver = member;
    }
    return false;
  }

  private static boolean isAssignmentTarget(Tree tree) {
    Tree parent = tree.parent();
    while (parent.is(Tree.Kind.PARENTHESIZED, Tree.Kind.TUPLE, Tree.Kind.LIST_LITERAL, Tree.Kind.EXPRESSION_LIST, Tree.Kind.UNPACKING_EXPR)) {
      tree = parent;
      parent = parent.parent();
    }
    return switch (parent) {
      case AssignmentStatement assignment -> assignment.lhsExpressions().contains(tree);
      case AnnotatedAssignment assignment -> assignment.variable() == tree && assignment.assignedValue() != null;
      case CompoundAssignmentStatement assignment -> assignment.lhsExpression() == tree;
      case ForStatement loop -> loop.expressions().contains(tree);
      case ComprehensionFor loop -> loop.loopExpression() == tree;
      case WithItem item -> item.expression() == tree;
      default -> false;
    };
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
