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

import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.checks.utils.GraphQLUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key="S6785")
public class GraphQLDenialOfServiceCheck extends PythonSubscriptionCheck {

  private static final Set<String> SAFE_VALIDATION_RULE_FQNS = Set.of("graphene.validation.DepthLimitValidator");

  private static final List<String> VALID_MIDDLEWARE_NAMES = List.of("DEPTH", "COST");

  private static final Predicate<String> VALID_MIDDLEWARE_PREDICATE = name ->
    VALID_MIDDLEWARE_NAMES.stream().anyMatch(mwName -> name.toUpperCase(Locale.ROOT).contains(mwName));
  private static final String MESSAGE_DEPTH = "Change this code to limit the depth of GraphQL queries.";
  private static final String MESSAGE_CIRCULAR = "This relationship creates circular references.";

  private final Set<Tree> circularReferences = new HashSet<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::searchForCircularReferences);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkGraphQLDepthLimit);
  }

  private void searchForCircularReferences(SubscriptionContext ctx) {
    circularReferences.clear();
    CircularRelationshipVisitor classVisitor = new CircularRelationshipVisitor(circularReferences);
    ctx.syntaxNode().accept(classVisitor);
  }

  private void checkGraphQLDepthLimit(SubscriptionContext ctx) {
    if (circularReferences.isEmpty()) {
      // Query depth should only be limited if there are circular references in the database
      return;
    }
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    List<Argument> arguments = callExpression.arguments();

    Optional.of(callExpression)
      .map(CallExpression::callee)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
      .filter(GraphQLUtils::isCallToAsView)
      .map(QualifiedExpression::qualifier)
      .filter(HasSymbol.class::isInstance)
      .map(HasSymbol.class::cast)
      .map(HasSymbol::symbol)
      .filter(GraphQLUtils::isOrExtendsGraphQLView)
      .filter(fqn -> !hasSafeValidationRules(arguments))
      .filter(fqn -> !hasSafeMiddlewares(arguments))
      .ifPresent(fqn -> {
        PreciseIssue preciseIssue = ctx.addIssue(callExpression.callee(), MESSAGE_DEPTH);
        circularReferences.forEach(circularRef -> preciseIssue.secondary(circularRef, MESSAGE_CIRCULAR));
      });
  }

  private static boolean hasSafeMiddlewares(List<Argument> arguments) {
    RegularArgument argument = TreeUtils.argumentByKeyword("middleware", arguments);
    if (argument == null) {
      return false;
    }

    return GraphQLUtils.extractListOrTupleArgumentValues(argument)
      .map(values -> GraphQLUtils.expressionsNameMatchPredicate(values, VALID_MIDDLEWARE_PREDICATE))
      .orElse(true);
  }

  private static boolean hasSafeValidationRules(List<Argument> arguments) {
    RegularArgument argument = TreeUtils.argumentByKeyword("validation_rules", arguments);
    if (argument == null) {
      return false;
    }

    return GraphQLUtils.extractListOrTupleArgumentValues(argument)
      .map(values -> (GraphQLUtils.expressionsNameMatchPredicate(values, VALID_MIDDLEWARE_PREDICATE)
          || GraphQLUtils.expressionsContainsSafeRuleFQN(values, SAFE_VALIDATION_RULE_FQNS::contains)))
      .orElse(true);
  }

  static class CircularRelationshipVisitor extends BaseTreeVisitor {

    final Set<Tree> circularReferences;

    public CircularRelationshipVisitor(Set<Tree> circularReferences) {
      this.circularReferences = circularReferences;
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      boolean createsCircularDependency = Optional.of(callExpression.callee())
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
        .filter(qualifiedExpression -> "relationship".equals(qualifiedExpression.name().name()))
        .map(QualifiedExpression::qualifier)
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
        .map(Expressions::singleAssignedNonNameValue)
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class))
        .map(CallExpression::callee)
        .filter(HasSymbol.class::isInstance)
        .map(HasSymbol.class::cast)
        .map(HasSymbol::symbol)
        .filter(s -> "flask_sqlalchemy.SQLAlchemy".equals(s.fullyQualifiedName()))
        .isPresent();
      if (createsCircularDependency) {
        circularReferences.add(callExpression);
      }
      super.visitCallExpression(callExpression);
    }
  }
}
