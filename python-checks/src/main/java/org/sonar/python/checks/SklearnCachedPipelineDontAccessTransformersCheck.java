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

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.tree.TupleImpl;

import static org.sonar.python.checks.utils.Expressions.getAssignedName;

@Rule(key = "S6971")
public class SklearnCachedPipelineDontAccessTransformersCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Avoid accessing transformers in a cached pipeline.";
  public static final String MESSAGE_SECONDARY = "The transformer is accessed here";
  public static final String MESSAGE_SECONDARY_CREATION = "The Pipeline is created here";
  private static final TypeMatcher IS_PIPELINE = TypeMatchers.isType("sklearn.pipeline.Pipeline");
  private static final TypeMatcher IS_MAKE_PIPELINE = TypeMatchers.isType("sklearn.pipeline.make_pipeline");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, SklearnCachedPipelineDontAccessTransformersCheck::checkCallExpr);
  }

  private static void checkCallExpr(SubscriptionContext subscriptionContext) {
    CallExpression callExpression = (CallExpression) subscriptionContext.syntaxNode();
    Optional<PipelineCreation> pipelineCreationOptional = isPipelineCreation(callExpression, subscriptionContext);
    if (pipelineCreationOptional.isEmpty()) {
      return;
    }
    PipelineCreation pipelineCreation = pipelineCreationOptional.get();

    var memoryArgument = TreeUtils.argumentByKeyword("memory", callExpression.arguments());
    if (memoryArgument == null || 
        memoryArgument.expression().is(Tree.Kind.NONE) || 
        memoryArgument.expression().typeV2() instanceof UnknownType) {
      return;
    }
    var stepsArgument = TreeUtils.nthArgumentOrKeyword(0, "steps", callExpression.arguments());

    StepsFromPipeline stepsFromPipeline = getStepsFromPipeline(stepsArgument, pipelineCreation);

    handleStepNames(subscriptionContext, stepsFromPipeline, pipelineCreation, callExpression);
  }

  private static StepsFromPipeline getStepsFromPipeline(@Nullable RegularArgument stepsArgument, PipelineCreation pipelineCreation) {
    Map<Name, String> nameToStepName = new HashMap<>();
    Optional<Expression> stepArgumentExpression = Optional.ofNullable(stepsArgument)
      .map(RegularArgument::expression);

    var stepNames = stepArgumentExpression.map(
      e -> pipelineCreation == PipelineCreation.PIPELINE ? extractFromPipeline(e, nameToStepName) : extractFromMakePipeline(e))
      .orElse(Stream.empty());
    return new StepsFromPipeline(nameToStepName, stepNames);
  }

  private record StepsFromPipeline(Map<Name, String> nameToStepName, Stream<Name> stepNames) {
  }

  private static void handleStepNames(SubscriptionContext subscriptionContext, StepsFromPipeline stepsFromPipeline, PipelineCreation pipelineCreation,
    CallExpression callExpression) {
    stepsFromPipeline.stepNames()
      .map(name -> Map.entry(name, symbolIsUsedInQualifiedExpression(name))).forEach(entry -> {
        Name name = entry.getKey();
        Map<Tree, QualifiedExpression> uses = entry.getValue();

        if (!uses.isEmpty()) {
          createIssue(subscriptionContext, stepsFromPipeline, pipelineCreation, callExpression, name, uses);
        }
      });
  }

  private static void createIssue(SubscriptionContext subscriptionContext, StepsFromPipeline stepsFromPipeline, PipelineCreation pipelineCreation, CallExpression callExpression,
    Name name, Map<Tree, QualifiedExpression> uses) {
    var issue = subscriptionContext.addIssue(name, MESSAGE);
    uses.forEach((useTree, qualExpr) -> issue.secondary(useTree, MESSAGE_SECONDARY));
    if (pipelineCreation == PipelineCreation.PIPELINE) {
      issue.secondary(callExpression.callee(), MESSAGE_SECONDARY_CREATION);
      uses
        .forEach(
          (useTree, qualExpr) -> getAssignedName(callExpression)
            .flatMap(pipelineBindingVariable -> getQuickFix(pipelineBindingVariable, name, qualExpr, stepsFromPipeline.nameToStepName()))
            .ifPresent(issue::addQuickFix));
    }
  }

  private static Stream<Name> extractFromMakePipeline(Expression stepArgumentExpression) {
    return Optional.of(stepArgumentExpression)
      .filter(e -> e.is(Tree.Kind.NAME))
      .map(Name.class::cast)
      .stream();
  }

  private static Stream<Name> extractFromPipeline(Expression stepArgumentExpression, Map<Name, String> nameToStepName) {
    return Optional.of(stepArgumentExpression)
      .filter(e -> e.is(Tree.Kind.LIST_LITERAL))
      .map(e -> ((ListLiteral) e).elements().expressions())
      .stream()
      .flatMap(Collection::stream)
      .filter(e -> e.is(Tree.Kind.TUPLE))
      .map(t -> ((TupleImpl) t).elements())
      .filter(e -> e.size() == 2)
      .filter(e -> e.get(1).is(Tree.Kind.NAME))
      .map(elements -> {
        if (elements.get(0).is(Tree.Kind.STRING_LITERAL)) {
          nameToStepName.put((Name) elements.get(1), ((StringLiteral) elements.get(0)).trimmedQuotesValue());
        }
        return elements;
      })
      .map(e -> e.get(1))
      .map(Name.class::cast);
  }

  private static Optional<PythonQuickFix> getQuickFix(Name pipelineBindingVariable, Tree name, QualifiedExpression qualifiedExpression, Map<Name, String> nameToStepName) {
    return Optional.ofNullable(nameToStepName.get(name))
      .map(stepName -> PythonQuickFix.newQuickFix("Replace the direct access to the transformer with an access to the `named_steps` attribute of the pipeline.")
        .addTextEdit(TextEditUtils.replace(qualifiedExpression.qualifier(), String.format("%s.named_steps[\"%s\"]", pipelineBindingVariable.name(), stepName)))
        .build());
  }

  private static Map<Tree, QualifiedExpression> symbolIsUsedInQualifiedExpression(Name name) {
    SymbolV2 symbol = name.symbolV2();
    if (symbol == null) {
      return new HashMap<>();
    }
    Map<Tree, QualifiedExpression> qualifiedExpressions = new HashMap<>();
    symbol.usages().stream()
      .filter(u -> u.tree().parent().is(Tree.Kind.QUALIFIED_EXPR))
      .forEach(u -> qualifiedExpressions.put(((QualifiedExpression) u.tree().parent()).qualifier(), ((QualifiedExpression) u.tree().parent())));

    return qualifiedExpressions;
  }

  private enum PipelineCreation {
    PIPELINE,
    MAKE_PIPELINE
  }

  private static Optional<PipelineCreation> isPipelineCreation(CallExpression callExpression, SubscriptionContext ctx) {
    if (IS_PIPELINE.isTrueFor(callExpression.callee(), ctx)) {
      return Optional.of(PipelineCreation.PIPELINE);
    }
    if (IS_MAKE_PIPELINE.isTrueFor(callExpression.callee(), ctx)) {
      return Optional.of(PipelineCreation.MAKE_PIPELINE);
    }
    return Optional.empty();
  }
}
