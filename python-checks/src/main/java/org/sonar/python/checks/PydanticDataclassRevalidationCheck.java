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
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8978")
public class PydanticDataclassRevalidationCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Explicitly set 'revalidate_instances' in this Pydantic model's configuration.";

  private static final TypeMatcher IS_PYDANTIC_MODEL = TypeMatchers.isOrExtendsType("pydantic.BaseModel");

  private static final TypeMatcher IS_CONFIG_DICT = TypeMatchers.isType("pydantic.ConfigDict");

  private static final TypeMatcher IS_DATACLASS_DECORATOR = TypeMatchers.isType("dataclasses.dataclass");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, PydanticDataclassRevalidationCheck::checkClassDef);
  }

  private static final String SECONDARY_MESSAGE = "The dataclass-typed field is defined here.";

  private static void checkClassDef(SubscriptionContext ctx) {
    ClassDef classDef = (ClassDef) ctx.syntaxNode();

    if (!IS_PYDANTIC_MODEL.isTrueFor(classDef.name(), ctx)) {
      return;
    }

    if (hasExplicitRevalidateInstances(classDef, ctx)
      || hasRevalidateInstancesKeywordArg(classDef)
      || hasInheritedRevalidateInstances(classDef, ctx)) {
      return;
    }

    List<AnnotatedAssignment> dataclassFields = classDef.body().statements().stream()
      .flatMap(TreeUtils.toStreamInstanceOfMapper(AnnotatedAssignment.class))
      .filter(aa -> isDataclassAnnotation(aa.annotation().expression(), ctx))
      .toList();

    if (!dataclassFields.isEmpty()) {
      var issue = ctx.addIssue(classDef.name(), MESSAGE);
      dataclassFields.forEach(aa -> issue.secondary(aa.variable(), SECONDARY_MESSAGE));
    }
  }

  /**
   * Returns true if any ancestor of this class (per Python's MRO) explicitly sets
   * {@code revalidate_instances} — either via {@code ConfigDict}/dict-literal assignment or as a
   * class keyword argument. Uses {@link ClassType#mro()} for correct C3-linearized traversal,
   * including diamond inheritance. Only ancestors whose {@code ClassDef} can be resolved in the
   * same file are inspected; cross-file ancestors are conservatively ignored.
   */
  private static boolean hasInheritedRevalidateInstances(ClassDef classDef, SubscriptionContext ctx) {
    PythonType type = classDef.name().typeV2();
    if (!(type instanceof ClassType classType)) {
      return false;
    }
    // Collect the full set of ancestor ClassTypes (skip self at index 0)
    Set<ClassType> ancestors = new HashSet<>(classType.mro().orElse(List.of()));
    ancestors.remove(classType);
    if (ancestors.isEmpty()) {
      return false;
    }
    // Walk base class Name expressions; for each one resolved to a same-file ClassDef,
    // check whether it (or any of its same-file ancestors) sets revalidate_instances.
    return collectSameFileClassDefs(classDef, ancestors).stream()
      .anyMatch(ancestorDef -> hasExplicitRevalidateInstances(ancestorDef, ctx)
        || hasRevalidateInstancesKeywordArg(ancestorDef));
  }

  /**
   * Collects {@link ClassDef} AST nodes for all ancestors of {@code classDef} that appear in
   * {@code ancestorTypes}, resolving transitively through same-file base classes.
   */
  private static List<ClassDef> collectSameFileClassDefs(ClassDef classDef, Set<ClassType> ancestorTypes) {
    ArgList argList = classDef.args();
    if (argList == null) {
      return List.of();
    }
    List<ClassDef> result = new ArrayList<>();
    for (Argument arg : argList.arguments()) {
      if (!(arg instanceof RegularArgument regArg) ||
        regArg.keywordArgument() != null ||
        !(regArg.expression() instanceof Name name)) {
        continue;
      }
      findClassDef(name.symbolV2()).ifPresent(baseClassDef -> {
        if (ancestorTypes.contains(baseClassDef.name().typeV2())) {
          result.add(baseClassDef);
          result.addAll(collectSameFileClassDefs(baseClassDef, ancestorTypes));
        }
      });
    }
    return result;
  }

  /**
   * Returns true if the class body contains a {@code ConfigDict(revalidate_instances=...)} assignment
   * with any value for {@code revalidate_instances}.
   * The variable name on the left-hand side is not checked; instead the callee type is matched.
   */
  private static boolean hasExplicitRevalidateInstances(ClassDef classDef, SubscriptionContext ctx) {
    return classDef.body().statements().stream()
      .flatMap(TreeUtils.toStreamInstanceOfMapper(AssignmentStatement.class))
      .anyMatch(stmt -> assignsConfigDictWithRevalidateInstances(stmt, ctx));
  }

  /**
   * Returns true if the class declaration uses a keyword argument for {@code revalidate_instances},
   * e.g. {@code class Foo(BaseModel, revalidate_instances='always')}.
   */
  private static boolean hasRevalidateInstancesKeywordArg(ClassDef classDef) {
    ArgList argList = classDef.args();
    if (argList == null) {
      return false;
    }
    return hasRevalidateInstancesKeyword(argList.arguments());
  }

  /**
   * Returns true if the assignment's right-hand side explicitly sets {@code revalidate_instances}.
   * Recognizes two forms:
   * <ul>
   *   <li>{@code ConfigDict(revalidate_instances=...)} — call identified by callee type</li>
   *   <li>{@code {'revalidate_instances': ...}} — plain dict literal</li>
   * </ul>
   * If the RHS is a variable reference, it is resolved to its single assigned value first.
   * This handles patterns like {@code COMMON_CONFIG = ConfigDict(revalidate_instances='always')}
   * used as {@code model_config = COMMON_CONFIG} (possibly at module level or via chained aliases).
   */
  private static boolean assignsConfigDictWithRevalidateInstances(AssignmentStatement stmt, SubscriptionContext ctx) {
    Expression rhs = Expressions.ifNameGetSingleAssignedNonNameValue(stmt.assignedValue()).orElse(null);
    if (rhs == null) {
      return false;
    }
    if (rhs instanceof CallExpression callExpr && IS_CONFIG_DICT.isTrueFor(callExpr.callee(), ctx)) {
      return hasRevalidateInstancesKeyword(callExpr.arguments());
    }
    if (rhs instanceof DictionaryLiteral dict) {
      return dict.elements().stream()
        .flatMap(TreeUtils.toStreamInstanceOfMapper(KeyValuePair.class))
        .map(KeyValuePair::key)
        .anyMatch(k -> k instanceof StringLiteral sl && "revalidate_instances".equals(sl.trimmedQuotesValue()));
    }
    return false;
  }

  private static boolean hasRevalidateInstancesKeyword(List<? extends Argument> arguments) {
    return arguments.stream()
      .flatMap(TreeUtils.toStreamInstanceOfMapper(RegularArgument.class))
      .anyMatch(arg -> {
        Name keyword = arg.keywordArgument();
        return keyword != null && "revalidate_instances".equals(keyword.name());
      });
  }

  /**
   * Returns true if the given annotation expression refers to a class decorated with {@code @dataclass}
   * that is defined in the same file.
   * Handles bare names, generic wrappers ({@code Optional[X]}, {@code List[X]}, {@code Annotated[X, ...]}, etc.)
   * and PEP-604 union syntax ({@code X | Y}).
   */
  private static boolean isDataclassAnnotation(Expression annotationExpr, SubscriptionContext ctx) {
    if (annotationExpr instanceof Name nameExpr) {
      return findClassDef(nameExpr.symbolV2())
        .map(referencedClassDef -> isDecoratedWithDataclass(referencedClassDef, ctx))
        .orElse(false);
    }
    if (annotationExpr instanceof SubscriptionExpression subscriptionExpr) {
      return subscriptionExpr.subscripts().expressions().stream()
        .anyMatch(inner -> isDataclassAnnotation(inner, ctx));
    }
    if (annotationExpr.is(Tree.Kind.BITWISE_OR)) {
      BinaryExpression binExpr = (BinaryExpression) annotationExpr;
      return isDataclassAnnotation(binExpr.leftOperand(), ctx)
        || isDataclassAnnotation(binExpr.rightOperand(), ctx);
    }
    return false;
  }

  private static Optional<ClassDef> findClassDef(@Nullable SymbolV2 symbolV2) {
    if (symbolV2 == null) {
      return Optional.empty();
    }
    return symbolV2.usages().stream()
      .filter(usage -> usage.kind() == UsageV2.Kind.CLASS_DECLARATION)
      .map(usage -> TreeUtils.firstAncestorOfKind(usage.tree(), Tree.Kind.CLASSDEF))
      .filter(Objects::nonNull)
      .map(ClassDef.class::cast)
      .findFirst();
  }

  private static boolean isDecoratedWithDataclass(ClassDef classDef, SubscriptionContext ctx) {
    return classDef.decorators().stream()
      .map(decorator -> decorator.expression() instanceof CallExpression callExpr ? callExpr.callee() : decorator.expression())
      .anyMatch(decoratorExpr -> IS_DATACLASS_DECORATOR.isTrueFor(decoratorExpr, ctx));
  }
}
