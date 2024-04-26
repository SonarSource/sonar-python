/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.Collection;
import java.util.HashSet;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.tree.TupleImpl;

@Rule(key = "S6971")
public class SklearnCachedPipelineDontAccessTransformersCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Avoid accessing transformers in a cached pipeline.";
  public static final String MESSAGE_SECONDARY = "Accessed here";
  public static final String MESSAGE_SECONDARY_CREATION = "Pipeline created here";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, SklearnCachedPipelineDontAccessTransformersCheck::checkCallExpr);
  }

  private static void checkCallExpr(SubscriptionContext subscriptionContext) {
    CallExpression callExpression = (CallExpression) subscriptionContext.syntaxNode();
    PipelineCreation pipelineCreation = isPipelineCreation(callExpression);
    if (pipelineCreation == PipelineCreation.NOT_PIPELINE) {
      return;
    }
    var memoryArgument = TreeUtils.argumentByKeyword("memory", callExpression.arguments());
    if (memoryArgument == null || memoryArgument.expression().is(Tree.Kind.NONE)) {
      return;
    }
    var stepsArgument = TreeUtils.nthArgumentOrKeyword(0, "steps", callExpression.arguments());

    Stream<Name> nameStream;
    Optional<Expression> stepArgumentExpression = Optional.ofNullable(stepsArgument)
      .map(RegularArgument::expression);

    if (pipelineCreation == PipelineCreation.PIPELINE) {
      var tuples = stepArgumentExpression
        .filter(e -> e.is(Tree.Kind.LIST_LITERAL))
        .map(e -> ((ListLiteral) e).elements().expressions())
        .stream()
        .flatMap(Collection::stream)
        .filter(e -> e.is(Tree.Kind.TUPLE));

      nameStream = tuples
        .map(t -> ((TupleImpl) t).elements())
        .filter(e -> e.size() == 2)
        .map(e -> e.get(1))
        .filter(e -> e.is(Tree.Kind.NAME))
        .map(Name.class::cast);
    } else {
      nameStream = stepArgumentExpression
        .filter(e -> e.is(Tree.Kind.NAME))
        .map(Name.class::cast)
        .stream();
    }
    var qualifiedUses = nameStream
      .collect(Collectors.toMap(n -> n, SklearnCachedPipelineDontAccessTransformersCheck::symbolIsUsedInQualifiedExpression));

    qualifiedUses.forEach((name, uses) -> {
      if (!uses.isEmpty()) {
        var issue = subscriptionContext.addIssue(name, MESSAGE);
        uses.forEach(use -> issue.secondary(use, MESSAGE_SECONDARY));
        if (pipelineCreation == PipelineCreation.PIPELINE) {
          issue.secondary(callExpression.callee(), MESSAGE_SECONDARY_CREATION);
        }
      }
    });

  }

  private static Set<Tree> symbolIsUsedInQualifiedExpression(Name name) {
    Symbol symbol = name.symbol();
    if (symbol == null) {
      return new HashSet<>();
    }
    Set<Tree> qualifiedExpressions = new HashSet<>();
    symbol.usages().stream()
      .filter(u -> u.tree().parent().is(Tree.Kind.QUALIFIED_EXPR))
      .forEach(u -> qualifiedExpressions.add(((QualifiedExpression) u.tree().parent()).qualifier()));
    return qualifiedExpressions;
  }

  private enum PipelineCreation {
    PIPELINE,
    MAKE_PIPELINE,
    NOT_PIPELINE
  }

  private static PipelineCreation isPipelineCreation(CallExpression callExpression) {
    return Optional.ofNullable(callExpression.calleeSymbol()).map(Symbol::fullyQualifiedName)
      .map(fqn -> {
        if ("sklearn.pipeline.Pipeline".equals(fqn)) {
          return PipelineCreation.PIPELINE;
        }
        if ("sklearn.pipeline.make_pipeline".equals(fqn)) {
          return PipelineCreation.MAKE_PIPELINE;
        }
        return PipelineCreation.NOT_PIPELINE;
      }).orElse(PipelineCreation.NOT_PIPELINE);
  }
}
