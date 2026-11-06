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

import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TryStatement;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.semantic.BuiltinSymbols;

// https://jira.sonarsource.com/browse/RSPEC-3984
@Rule(key = "S3984")
public class ExceptionNotThrownCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Raise this exception or remove this useless statement.";
  public static final String QUICK_FIX_MESSAGE = "Raise this exception";
  private static final Set<String> BUILTIN_EXCEPTIONS_FQNS = Stream.of(
    BuiltinSymbols.EXCEPTIONS,
    BuiltinSymbols.EXCEPTIONS_PYTHON2).flatMap(Set::stream).filter(n -> !"WindowsError".equals(n)).collect(Collectors.toSet());
  private static final Set<String> AVAILABILITY_PROBE_CATCHERS = Set.of("NameError", "ImportError", "Exception", "BaseException");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ExceptionNotThrownCheck::check);
    context.registerSyntaxNodeConsumer(Tree.Kind.NAME, ExceptionNotThrownCheck::check);
  }

  private static void check(SubscriptionContext subscriptionContext) {
    Expression expression = (Expression) subscriptionContext.syntaxNode();
    PythonType classCandidate = expression instanceof CallExpression call ? call.callee().typeV2() : expression.typeV2();
    if (!isThrowable(classCandidate) || !expression.parent().is(Tree.Kind.EXPRESSION_STMT) || isExceptionAvailabilityCheck(expression)) {
      return;
    }

    var issue = subscriptionContext.addIssue(expression, MESSAGE);
    var quickFix = PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
      .addTextEdit(TextEditUtils.insertBefore(expression, "raise "))
      .build();
    issue.addQuickFix(quickFix);
  }

  private static boolean isThrowable(PythonType type) {
    if (!(type instanceof ClassType classType)) {
      return false;
    }
    return isBuiltinException(classType) || classType.superClasses().stream()
      .map(TypeWrapper::type)
      .filter(ClassType.class::isInstance)
      .map(ClassType.class::cast)
      .anyMatch(ExceptionNotThrownCheck::isBuiltinException);
  }

  private static boolean isBuiltinException(ClassType classType) {
    return BUILTIN_EXCEPTIONS_FQNS.contains(classType.fullyQualifiedName());
  }

  private static boolean isExceptionAvailabilityCheck(Expression expression) {
    if (!(expression instanceof Name name)) {
      return false;
    }
    Tree child = expression;
    for (Tree parent = expression.parent(); parent != null; child = parent, parent = parent.parent()) {
      if (parent.is(Tree.Kind.FUNCDEF, Tree.Kind.LAMBDA)) {
        return false;
      }
      if (parent instanceof TryStatement tryStatement && child == tryStatement.body()) {
        return tryStatement.exceptClauses().stream()
          .anyMatch(exceptClause -> catchesAvailabilityProbeException(exceptClause) && assignsName(exceptClause, name));
      }
    }
    return false;
  }

  private static boolean catchesAvailabilityProbeException(ExceptClause exceptClause) {
    Expression exception = exceptClause.exception();
    if (exception == null) {
      return true;
    }
    return catchesAvailabilityProbeException(exception);
  }

  private static boolean catchesAvailabilityProbeException(Expression exception) {
    if (exception instanceof Tuple tuple) {
      return tuple.elements().stream().anyMatch(ExceptionNotThrownCheck::catchesAvailabilityProbeException);
    }
    if (exception instanceof Name name) {
      return AVAILABILITY_PROBE_CATCHERS.contains(name.name());
    }
    return exception instanceof QualifiedExpression qualifiedExpression
      && qualifiedExpression.qualifier() instanceof Name qualifier
      && "builtins".equals(qualifier.name())
      && AVAILABILITY_PROBE_CATCHERS.contains(qualifiedExpression.name().name());
  }

  private static boolean assignsName(ExceptClause exceptClause, Name probed) {
    return assignsName(exceptClause.body(), probed);
  }

  private static boolean assignsName(Tree tree, Name probed) {
    if (tree.is(Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF, Tree.Kind.LAMBDA)) {
      return false;
    }
    var probedSymbol = probed.symbolV2();
    if (tree instanceof Name candidate && probedSymbol != null && candidate.symbolV2() == probedSymbol) {
      return probedSymbol.usages().stream()
        .anyMatch(usage -> usage.tree() == candidate && usage.isBindingUsage());
    }
    return tree.children().stream().anyMatch(child -> assignsName(child, probed));
  }

}
