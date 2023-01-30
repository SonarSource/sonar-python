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
package org.sonar.python.checks.cdk;

import java.util.Deque;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.DictionaryLiteralElement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnpackingExpression;
import org.sonar.python.checks.Expressions;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.checks.cdk.CdkPredicate.isListLiteral;
import static org.sonar.python.checks.cdk.CdkPredicate.isString;

public class CdkUtils {

  private CdkUtils() {
  }

  public static Optional<Integer> getInt(Expression expression) {
    try {
      return Optional.of((int)((NumericLiteral) expression).valueAsLong());
    } catch (ClassCastException e) {
      return Optional.empty();
    }
  }

  public static Optional<String> getString(Expression expression) {
    try {
      return Optional.of(((StringLiteral) expression).trimmedQuotesValue());
    } catch (ClassCastException e) {
      return Optional.empty();
    }
  }

  public static Optional<CallExpression> getCall(Expression expression, String fqn) {
    if (expression.is(Tree.Kind.CALL_EXPR) && CdkPredicate.isFqn(fqn).test(expression)) {
      return Optional.of((CallExpression) expression);
    }
    return Optional.empty();
  }

  public static Optional<ListLiteral> getListExpression(ExpressionFlow expression) {
    return expression.getExpression(isListLiteral()).map(ListLiteral.class::cast);
  }

  public static Optional<DictionaryLiteral> getDictionary(Expression expression) {
    if (expression.is(Tree.Kind.DICTIONARY_LITERAL)) {
      return Optional.of((DictionaryLiteral) expression);
    }
    return Optional.empty();
  }

  /**
   * Resolve a particular argument of a call by keyword or get an empty optional if the argument is not set nor resolvable.
   */
  protected static Optional<ExpressionFlow> getArgument(SubscriptionContext ctx, CallExpression callExpression, String argumentName) {
    return getArgument(ctx, callExpression, TreeUtils.argumentByKeyword(argumentName,
      callExpression.arguments()), argumentName);
  }

  protected static Optional<ExpressionFlow> getArgument(CallExpression callExpression, String argumentName) {
    // ctx not required as long as we don't try to raise an issue with ExpressionFlow
    return getArgument(null, callExpression, TreeUtils.argumentByKeyword(argumentName, callExpression.arguments()), argumentName);
  }

  private static Optional<ExpressionFlow> getArgument(@Nullable SubscriptionContext ctx, CallExpression callExpression, @Nullable RegularArgument regArg, String argumentName) {
    List<Argument> arguments = callExpression.arguments();
    Optional<ExpressionFlow> argument = Optional.ofNullable(regArg)
      .map(RegularArgument::expression)
      .map(expression -> ExpressionFlow.build(ctx, expression));

    if (argument.isEmpty()) {
      return arguments.stream()
        .filter(UnpackingExpression.class::isInstance)
        .map(UnpackingExpression.class::cast)
        .map(unpacking -> resolveValue(unpacking.expression(), argumentName, new HashSet<>())
          .map(exp -> new ExpressionFlow(ctx, new LinkedList<>(List.of(unpacking.expression(), exp))))
          .orElseGet(() -> new UnresolvedExpressionFlow(ctx)))
        .findFirst();
    }
    return argument;
  }

  private static Optional<Expression> resolveValue(@Nullable Expression expression, String argumentName, Set<Expression> visited) {
    if (expression == null || visited.contains(expression)) {
      return Optional.empty();
    }
    visited.add(expression);

    if (expression.is(Tree.Kind.NAME)) {
      var assignedValue = Expressions.singleAssignedValue((Name) expression);
      var resolved = resolveValue(assignedValue, argumentName, visited);
      return resolved.isEmpty() ? Optional.of(expression) : resolved;
    } else if (expression.is(Tree.Kind.DICTIONARY_LITERAL)) {
      return ((DictionaryLiteral) expression).elements()
        .stream()
        .filter(KeyValuePair.class::isInstance)
        .map(KeyValuePair.class::cast)
        .filter(el -> el.key() instanceof StringLiteral)
        .filter(el -> Objects.equals(argumentName, ((StringLiteral) el.key()).trimmedQuotesValue()))
        .map(KeyValuePair::value)
        .map(value -> resolveValue(value, argumentName, visited))
        .findFirst()
        .orElseGet(Optional::empty);
    }
    return Optional.empty();
  }

  /**
   * Resolve a particular argument of a call by keyword or position. Return an empty optional if the argument is not set nor resolvable.
   */
  public static Optional<ExpressionFlow> getArgument(SubscriptionContext ctx, CallExpression callExpression, String argumentName, int argumentPosition) {
    return Optional.ofNullable(TreeUtils.nthArgumentOrKeyword(argumentPosition, argumentName, callExpression.arguments()))
      .map(argument -> ExpressionFlow.build(ctx, argument.expression()));
  }

  /**
   * Returns a ListLiteral if the given expression flow origins of this kind
   */
  public static Optional<ListLiteral> getList(ExpressionFlow flow) {
    return flow.getExpression(e -> e.is(Tree.Kind.LIST_LITERAL))
      .map(ListLiteral.class::cast);
  }

  /**
   * Creates flows for the individual elements of a list
   */
  public static List<CdkUtils.ExpressionFlow> getListElements(SubscriptionContext ctx, ListLiteral list) {
    return list.elements().expressions().stream()
      .map(expression -> CdkUtils.ExpressionFlow.build(ctx, expression))
      .collect(Collectors.toList());
  }

  /**
   * Returns a DictionaryLiteral if the given expression flow origins of this kind
   */
  public static Optional<DictionaryLiteral> getDictionary(ExpressionFlow flow) {
    return flow.getExpression(e -> e.is(Tree.Kind.DICTIONARY_LITERAL))
      .map(DictionaryLiteral.class::cast);
  }

  /**
   * By resolving the individual dictionary elements, a key-value pair can be returned by a given key. The value is also a resolved flow.
   */
  public static Optional<ResolvedKeyValuePair> getDictionaryPair(SubscriptionContext ctx, DictionaryLiteral dict, String key) {
    return getDictionaryPair(CdkUtils.resolveDictionary(ctx, dict), key);
  }

  /**
   * A key-value pair can be returned by a given key. The value is also a resolved flow.
   */
  public static Optional<ResolvedKeyValuePair> getDictionaryPair(List<ResolvedKeyValuePair> pairs, String key) {
    return pairs.stream()
      .filter(pair -> pair.key.hasExpression(isString(key)))
      .findFirst();
  }

  /**
   * Return a resolved dictionary value by a given key
   */
  public static Optional<ExpressionFlow> getDictionaryValue(List<ResolvedKeyValuePair> pairs, String key) {
    return getDictionaryPair(pairs, key)
      .map(pair -> pair.value);
  }

  /**
   * Collects all dictionary elements of a list as a return.
   */
  public static List<DictionaryLiteral> getDictionaryInList(SubscriptionContext ctx, ListLiteral listeners) {
    return getListElements(ctx, listeners).stream()
      .map(CdkUtils::getDictionary)
      .flatMap(Optional::stream)
      .collect(Collectors.toList());
  }

  /**
   * Resolves all elements of a dictionary. All keys and values are resolved into flows.
   */
  public static List<ResolvedKeyValuePair> resolveDictionary(SubscriptionContext ctx, DictionaryLiteral dict) {
    return dict.elements().stream()
      .map(e -> CdkUtils.getKeyValuePair(ctx, e))
      .flatMap(Optional::stream)
      .collect(Collectors.toUnmodifiableList());
  }

  /**
   * Resolve the key and value of a dictionary element or get an empty optional if the element is an UnpackingExpression.
   */
  public static Optional<ResolvedKeyValuePair> getKeyValuePair(SubscriptionContext ctx, DictionaryLiteralElement element) {
    return element.is(Tree.Kind.KEY_VALUE_PAIR) ? Optional.of(ResolvedKeyValuePair.build(ctx, (KeyValuePair) element)) : Optional.empty();
  }

  /**
   * The expression flow represents the propagation of an expression.
   * It serves as a resolution path from the use of the expression up to the value assignment.
   * For example, if the value of an argument expression did not occur directly in the function call, the value can be tracked back.
   * The flow allows on the one hand to check the assigned value
   * and on the other hand to display the assignment path of the relevant value when creating an issue.
   */
  static class ExpressionFlow {

    private static final String TAIL_MESSAGE = "Propagated setting.";

    private final SubscriptionContext ctx;
    private final Deque<Expression> locations;

    private ExpressionFlow(SubscriptionContext ctx, Deque<Expression> locations) {
      this.ctx = ctx;
      this.locations = locations;
    }

    protected static ExpressionFlow build(SubscriptionContext ctx, Expression expression) {
      Deque<Expression> locations = new LinkedList<>();
      resolveLocations(expression, locations);
      return new ExpressionFlow(ctx, locations);
    }

    static void resolveLocations(Expression expression, Deque<Expression> locations) {
      locations.add(expression);
      if (expression.is(Tree.Kind.NAME)) {
        Expression singleAssignedValue = Expressions.singleAssignedValue(((Name) expression));
        if (singleAssignedValue != null && !locations.contains(singleAssignedValue)) {
          resolveLocations(singleAssignedValue, locations);
        }
      }
    }

    public void addIssue(String primaryMessage, IssueLocation... secondaryLocations) {
      PythonCheck.PreciseIssue issue = ctx.addIssue(locations.getFirst().parent(), primaryMessage);
      locations.stream().skip(1).forEach(expression -> issue.secondary(expression.parent(), TAIL_MESSAGE));
      Stream.of(secondaryLocations).forEach(issue::secondary);
    }

    public void addIssueIf(Predicate<Expression> predicate, String primaryMessage, IssueLocation... secondaryLocations) {
      if (hasExpression(predicate)) {
        addIssue(primaryMessage, secondaryLocations);
      }
    }

    public void addIssueIf(Predicate<Expression> predicate, String primaryMessage, CallExpression call) {
      if (hasExpression(predicate)) {
        ctx.addIssue(call.callee(), primaryMessage);
      }
    }

    public boolean hasExpression(Predicate<Expression> predicate) {
      return locations.stream().anyMatch(predicate);
    }

    public Optional<Expression> getExpression(Predicate<Expression> predicate) {
      return locations.stream().filter(predicate).findFirst();
    }

    public Deque<Expression> locations() {
      return locations;
    }

    public Expression getLast() {
      return locations().getLast();
    }

    public IssueLocation asSecondaryLocation(String message) {
      return IssueLocation.preciseLocation(getLast().parent(), message);
    }

    public SubscriptionContext ctx() {
      return ctx;
    }
  }

  /**
   * In the case of unpacking expression, we cannot generate flows at the moment.
   * However, to avoid a wrong interpretation of the unpacked expression in the context of absent arguments,
   * an alternative dummy must be returned, which should not lead to false positives.
   * The resolving of such expressions can be improved by <a href="https://sonarsource.atlassian.net/browse/SONARPY-1164">SONARPY-1164</a> if necessary.
   */
  static class UnresolvedExpressionFlow extends ExpressionFlow {

    private UnresolvedExpressionFlow(SubscriptionContext ctx) {
      super(ctx, new LinkedList<>());
    }
  }

  /**
   * Dataclass to store a resolved KeyValuePair structure
   */
  static class ResolvedKeyValuePair {

    final ExpressionFlow key;
    final ExpressionFlow value;

    private ResolvedKeyValuePair(ExpressionFlow key, ExpressionFlow value) {
      this.key = key;
      this.value = value;
    }

    static ResolvedKeyValuePair build(SubscriptionContext ctx, KeyValuePair pair) {
      return new ResolvedKeyValuePair(ExpressionFlow.build(ctx, pair.key()), ExpressionFlow.build(ctx, pair.value()));
    }
  }
}
