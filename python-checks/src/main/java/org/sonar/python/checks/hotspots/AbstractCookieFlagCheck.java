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
package org.sonar.python.checks.hotspots;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.checks.Expressions.isFalsy;

public abstract class AbstractCookieFlagCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.ASSIGNMENT_STMT, ctx -> {
      AssignmentStatement assignment = (AssignmentStatement) ctx.syntaxNode();
      getSubscriptionToCookies(assignment.lhsExpressions())
        .forEach(sub -> {
          if (isSettingFlag(sub, flagName()) && isFalsy(assignment.assignedValue())) {
            ctx.addIssue(assignment, message());
          }
        });
    });

    context.registerSyntaxNodeConsumer(Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      verifyCallExpression(ctx, callExpression);
    });
  }

  private void verifyCallExpression(SubscriptionContext ctx, CallExpression callExpression) {
    if (callExpression.arguments().stream().anyMatch(argument -> argument.is(Kind.UNPACKING_EXPR))) {
      return;
    }
    var registry = methodArgumentsToCheckRegistry();

    var methodName = getCallExpressionMethodName(callExpression);
    var methodFqn = getCallExpressionMethodFqn(callExpression);
    if ((methodName != null && registry.hasMethodName(methodName)) || (methodFqn != null && registry.hasMethodFqn(methodFqn))) {
      findMethodArgumentToCheck(callExpression)
        .ifPresent(methodArgumentsToCheck -> {
          RegularArgument argument = TreeUtils.nthArgumentOrKeyword(methodArgumentsToCheck.argumentPosition(), methodArgumentsToCheck.argumentName(), callExpression.arguments());
          if ((methodArgumentsToCheck.complainIfMissing() && argument == null)
            || methodArgumentsToCheck.invalidArgumentPredicate().test(argument)) {
            ctx.addIssue(callExpression.callee(), message());
          }
        });
    }
  }

  private Optional<MethodArgumentsToCheck> findMethodArgumentToCheck(CallExpression callExpression) {
    return Optional.of(callExpression)
      .map(AbstractCookieFlagCheck::getCallExpressionMethodFqn)
      .map(methodArgumentsToCheckRegistry()::getByMethodFqn)
      .or(() -> Optional.of(callExpression)
          .map(AbstractCookieFlagCheck::getCallExpressionMethodName)
          .map(methodArgumentsToCheckRegistry()::getByMethodName)
          .stream()
          .flatMap(Collection::stream)
          .filter(methodArgumentsToCheck -> canBeOrExtendMatches(callExpression, methodArgumentsToCheck))
          .findFirst()
      );
  }

  private static Boolean canBeOrExtendMatches(CallExpression callExpression, MethodArgumentsToCheck methodArgumentsToCheck) {
    return Optional.of(callExpression)
      .map(CallExpression::callee)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
      .map(QualifiedExpression::qualifier)
      .map(Expression::type)
      .map(t -> t.mustBeOrExtend(methodArgumentsToCheck.calleeFqn()))
      .orElse(false);
  }

  private static String getCallExpressionMethodName(CallExpression callExpression) {
    return Optional.of(callExpression)
      .map(CallExpression::calleeSymbol)
      .map(Symbol::name)
      .orElse(null);
  }

  private static String getCallExpressionMethodFqn(CallExpression callExpression) {
    return Optional.of(callExpression)
      .map(CallExpression::calleeSymbol)
      .map(Symbol::fullyQualifiedName)
      .orElse(null);
  }

  private static Stream<SubscriptionExpression> getSubscriptionToCookies(List<ExpressionList> lhsExpressions) {
    return lhsExpressions.stream()
      .flatMap(expressionList -> expressionList.expressions().stream())
      .filter(lhs -> {
        if (lhs.is(Kind.SUBSCRIPTION)) {
          SubscriptionExpression sub = (SubscriptionExpression) lhs;
          return getObject(sub.object()).type().canOnlyBe("http.cookies.SimpleCookie");
        }
        return false;
      })
      .map(SubscriptionExpression.class::cast);
  }

  private static boolean isSettingFlag(SubscriptionExpression sub, String flagName) {
    List<ExpressionList> subscripts = getSubscripts(sub);
    if (subscripts.size() == 1) {
      return false;
    }
    return subscripts.stream()
      .skip(1)
      .anyMatch(s -> s.expressions().size() == 1 && isFlagNameStringLiteral(s.expressions().get(0), flagName));
  }

  private static List<ExpressionList> getSubscripts(SubscriptionExpression sub) {
    Deque<ExpressionList> subscripts = new ArrayDeque<>();
    subscripts.addFirst(sub.subscripts());
    Expression object = sub.object();
    while (object.is(Kind.SUBSCRIPTION)) {
      subscripts.addFirst(((SubscriptionExpression) object).subscripts());
      object = ((SubscriptionExpression) object).object();
    }
    return new ArrayList<>(subscripts);
  }

  private static boolean isFlagNameStringLiteral(Expression expression, String value) {
    return expression.is(Kind.STRING_LITERAL) && ((StringLiteral) expression).trimmedQuotesValue().equalsIgnoreCase(value);
  }

  private static Expression getObject(Expression object) {
    if (object.is(Kind.SUBSCRIPTION)) {
      return getObject(((SubscriptionExpression) object).object());
    }
    return object;
  }

  abstract String flagName();

  abstract String message();

  abstract MethodArgumentsToCheckRegistry methodArgumentsToCheckRegistry();

  static class MethodArgumentsToCheckRegistry {
    private final Map<String, List<MethodArgumentsToCheck>> byMethodName;
    private final Map<String, MethodArgumentsToCheck> byMethodFqn;

    public MethodArgumentsToCheckRegistry(MethodArgumentsToCheck... arguments) {
      byMethodName = Stream.of(arguments)
        .filter(argument -> Objects.nonNull(argument.methodName))
        .collect(Collectors.groupingBy(MethodArgumentsToCheck::methodName));

      byMethodFqn = Stream.of(arguments)
        .collect(Collectors.toMap(MethodArgumentsToCheck::methodFqn, Function.identity(), (v1, v2) -> v2));
    }

    List<MethodArgumentsToCheck> getByMethodName(String methodName) {
      return byMethodName.get(methodName);
    }

    MethodArgumentsToCheck getByMethodFqn(String methodFqn) {
      return byMethodFqn.get(methodFqn);
    }

    boolean hasMethodName(String methodName) {
      return byMethodName.containsKey(methodName);
    }

    boolean hasMethodFqn(String methodFqn) {
      return byMethodFqn.containsKey(methodFqn);
    }
  }

  static class MethodArgumentsToCheck {
    private final String calleeFqn;
    private final String methodName;
    private final String methodFqn;
    private final String argumentName;
    private final int argumentPosition;
    private final boolean complainIfMissing;
    private final Predicate<RegularArgument> invalidArgumentPredicate;


    public MethodArgumentsToCheck(String calleeFqn, String argumentName, int argumentPosition, boolean complainIfMissing,
      Predicate<RegularArgument> invalidArgumentPredicate) {
      this(calleeFqn, null, argumentName, argumentPosition, complainIfMissing, invalidArgumentPredicate);
    }

    public MethodArgumentsToCheck(String calleeFqn, String methodName, String argumentName, int argumentPosition, boolean complainIfMissing) {
      this(calleeFqn, methodName, argumentName, argumentPosition, complainIfMissing, arg -> isFalsy(arg.expression()));
    }

    public MethodArgumentsToCheck(String calleeFqn, @Nullable String methodName, String argumentName, int argumentPosition,
      boolean complainIfMissing, Predicate<RegularArgument> invalidArgumentPredicate) {
      this.calleeFqn = calleeFqn;
      this.methodName = methodName;
      this.invalidArgumentPredicate = invalidArgumentPredicate;
      methodFqn = Optional.ofNullable(methodName)
        .map(mn -> calleeFqn + "." + mn)
        .orElse(calleeFqn);
      this.argumentName = argumentName;
      this.argumentPosition = argumentPosition;
      this.complainIfMissing = complainIfMissing;
    }

    public String calleeFqn() {
      return calleeFqn;
    }

    public String methodName() {
      return methodName;
    }

    public String argumentName() {
      return argumentName;
    }

    public int argumentPosition() {
      return argumentPosition;
    }

    public boolean complainIfMissing() {
      return complainIfMissing;
    }

    public String methodFqn() {
      return methodFqn;
    }

    public Predicate<RegularArgument> invalidArgumentPredicate() {
      return invalidArgumentPredicate;
    }
  }
}
