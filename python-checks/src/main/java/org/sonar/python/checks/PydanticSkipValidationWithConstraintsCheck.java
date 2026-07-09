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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8971")
public class PydanticSkipValidationWithConstraintsCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Remove either \"SkipValidation\" or the validation constraints from this annotation.";

  private static final TypeMatcher IS_PYDANTIC_MODEL = TypeMatchers.isOrExtendsType("pydantic.BaseModel");

  private static final TypeMatcher IS_TYPING_ANNOTATED = TypeMatchers.any(
    TypeMatchers.isType("typing.Annotated"),
    TypeMatchers.isType("typing_extensions.Annotated"));

  private static final TypeMatcher IS_SKIP_VALIDATION = TypeMatchers.any(
    TypeMatchers.withFQN("pydantic.SkipValidation"),
    TypeMatchers.withFQN("pydantic.functional_validators.SkipValidation"));

  private static final TypeMatcher IS_PYDANTIC_FIELD = TypeMatchers.isType("pydantic.fields.Field");

  /**
   * Keyword arguments of pydantic.Field() that impose actual value constraints on field inputs.
   * Only pure value-bound constraints are included: numeric bounds, length limits, and pattern.
   * Arguments such as default, alias, description, title, etc. are metadata-only.
   * Behavioral flags are intentionally excluded:
   * - strict, allow_inf_nan, coerce_numbers_to_str, fail_fast: control how validation runs,
   *   but with SkipValidation no validation runs at all, so these have no effect.
   * - discriminator: structural union routing metadata, not a value constraint.
   */
  private static final Set<String> FIELD_VALIDATION_CONSTRAINT_ARGS = Set.of(
    "gt", "ge", "lt", "le",
    "multiple_of",
    "min_length", "max_length",
    "pattern",
    "max_digits", "decimal_places");

  private static final TypeMatcher IS_VALIDATION_CONSTRAINT = TypeMatchers.any(
    TypeMatchers.withFQN("pydantic.StringConstraints"),
    TypeMatchers.withFQN("pydantic.types.StringConstraints"),
    TypeMatchers.withFQN("pydantic.AfterValidator"),
    TypeMatchers.withFQN("pydantic.functional_validators.AfterValidator"),
    TypeMatchers.withFQN("pydantic.BeforeValidator"),
    TypeMatchers.withFQN("pydantic.functional_validators.BeforeValidator"),
    TypeMatchers.withFQN("pydantic.PlainValidator"),
    TypeMatchers.withFQN("pydantic.functional_validators.PlainValidator"),
    TypeMatchers.withFQN("pydantic.WrapValidator"),
    TypeMatchers.withFQN("pydantic.functional_validators.WrapValidator"),
    TypeMatchers.withFQN("pydantic.WrapSerializer"),
    TypeMatchers.withFQN("pydantic.functional_serializers.WrapSerializer"),
    TypeMatchers.withFQN("pydantic.PlainSerializer"),
    TypeMatchers.withFQN("pydantic.functional_serializers.PlainSerializer"));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, PydanticSkipValidationWithConstraintsCheck::checkClassDef);
  }

  private static void checkClassDef(SubscriptionContext ctx) {
    ClassDef classDef = (ClassDef) ctx.syntaxNode();

    if (!IS_PYDANTIC_MODEL.isTrueFor(classDef.name(), ctx)) {
      return;
    }

    classDef.body().statements().stream()
      .flatMap(TreeUtils.toStreamInstanceOfMapper(AnnotatedAssignment.class))
      .forEach(annotatedAssignment -> checkFieldAnnotation(ctx, annotatedAssignment));
  }

  private static void checkFieldAnnotation(SubscriptionContext ctx, AnnotatedAssignment annotatedAssignment) {
    TypeAnnotation annotation = annotatedAssignment.annotation();
    Expression annotationExpr = annotation.expression();

    if (!(annotationExpr instanceof SubscriptionExpression subscriptionExpr) ||
      !IS_TYPING_ANNOTATED.isTrueFor(subscriptionExpr.object(), ctx)) {
      return;
    }

    // Collect all subscript elements from this Annotated and any nested Annotated
    List<Expression> allElements = collectAnnotatedElements(subscriptionExpr, ctx);

    Expression skipValidationExpr = allElements.stream()
      .filter(expr -> isSkipValidation(expr, ctx))
      .findFirst()
      .orElse(null);

    if (skipValidationExpr == null) {
      return;
    }

    List<Expression> constraintSecondaryExprs = allElements.stream()
      .map(expr -> resolveConstraintSecondary(expr, ctx))
      .filter(Optional::isPresent)
      .map(Optional::get)
      .toList();

    if (!constraintSecondaryExprs.isEmpty()) {
      var issue = ctx.addIssue(skipValidationExpr, MESSAGE);
      constraintSecondaryExprs.forEach(expr -> issue.secondary(expr, "Validation constraint is set here."));
    }
  }

  /**
   * Recursively collects all non-type elements from Annotated[...] subscripts.
   * The first element in Annotated is the actual type (skipped). All subsequent
   * elements are metadata/constraints. Nested Annotated[...] are unwrapped.
   */
  private static List<Expression> collectAnnotatedElements(SubscriptionExpression annotatedExpr, SubscriptionContext ctx) {
    List<Expression> elements = new ArrayList<>();
    List<Expression> subscriptExprs = annotatedExpr.subscripts().expressions();

    // subscriptExprs[0] is the annotated type, subscriptExprs[1..] are metadata
    for (int i = 1; i < subscriptExprs.size(); i++) {
      Expression element = subscriptExprs.get(i);
      // If this element is itself an Annotated[...], flatten it recursively
      if (element instanceof SubscriptionExpression nestedSubscription
        && IS_TYPING_ANNOTATED.isTrueFor(nestedSubscription.object(), ctx)) {
        elements.addAll(collectAnnotatedElements(nestedSubscription, ctx));
      } else {
        elements.add(element);
      }
    }

    // Also recurse into the type argument if it is a nested Annotated
    if (!subscriptExprs.isEmpty()) {
      Expression typeArg = subscriptExprs.get(0);
      if (typeArg instanceof SubscriptionExpression nestedAnnotated
        && IS_TYPING_ANNOTATED.isTrueFor(nestedAnnotated.object(), ctx)) {
        elements.addAll(collectAnnotatedElements(nestedAnnotated, ctx));
      }
    }

    return elements;
  }

  private static boolean isSkipValidation(Expression expr, SubscriptionContext ctx) {
    // SkipValidation can appear as a bare name or as SkipValidation[T] subscription
    if (expr instanceof SubscriptionExpression subscriptionExpr) {
      return IS_SKIP_VALIDATION.isTrueFor(subscriptionExpr.object(), ctx);
    }
    return IS_SKIP_VALIDATION.isTrueFor(expr, ctx);
  }

  /**
   * Returns the expression to use as the secondary location if the given expression is a validation
   * constraint (or a name that resolves to one), or empty otherwise.
   *
   * <p>For direct constraint expressions (e.g. {@code Field(gt=0)}, {@code AfterValidator(...)}),
   * the expression itself is returned. For bare name references (e.g. {@code constraint} where
   * {@code constraint = Field(gt=0)} is assigned in the class body), the single assigned value is
   * returned so the secondary location points at the constraint definition, not the reference.</p>
   */
  private static Optional<Expression> resolveConstraintSecondary(Expression expr, SubscriptionContext ctx) {
    if (isDirectConstraint(expr, ctx)) {
      return Optional.of(expr);
    }
    // If expr is a bare name, check whether it was assigned a single constraint value.
    if (expr instanceof Name name) {
      Expression assigned = Expressions.singleAssignedValue(name);
      if (assigned != null && isDirectConstraint(assigned, ctx)) {
        return Optional.of(assigned);
      }
    }
    return Optional.empty();
  }

  private static boolean isDirectConstraint(Expression expr, SubscriptionContext ctx) {
    // Constraints typically appear as CallExpression: Field(gt=0), StringConstraints(...)
    if (expr instanceof CallExpression callExpr) {
      // Non-Field constraints (validators, serializers, StringConstraints): always count
      if (IS_VALIDATION_CONSTRAINT.isTrueFor(callExpr.callee(), ctx)) {
        return true;
      }
      // Field(): only count as a constraint when at least one validation-specific kwarg is present.
      // Calls like Field(default=0), Field(alias="x"), Field(description="...") are metadata-only
      // and should not be flagged when combined with SkipValidation.
      if (IS_PYDANTIC_FIELD.isTrueFor(callExpr.callee(), ctx)) {
        return hasFieldValidationConstraintArg(callExpr);
      }
    }
    // Some validators can be used as bare references too
    return IS_VALIDATION_CONSTRAINT.isTrueFor(expr, ctx);
  }

  private static boolean hasFieldValidationConstraintArg(CallExpression callExpr) {
    return callExpr.arguments().stream()
      .filter(RegularArgument.class::isInstance)
      .map(arg -> ((RegularArgument) arg).keywordArgument())
      .filter(Objects::nonNull)
      .map(Name::name)
      .anyMatch(FIELD_VALIDATION_CONSTRAINT_ARGS::contains);
  }
}
