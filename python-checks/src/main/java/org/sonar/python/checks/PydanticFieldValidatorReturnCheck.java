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

import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.cfg.CfgBranchingBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.WhileStatement;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.checks.utils.Expressions;

/**
 * Detects Pydantic {@code @field_validator} methods that have a code path without an explicit return value.
 * When a field validator does not return a value, Pydantic silently stores {@code None} as the field value,
 * violating the field's type contract without raising any error.
 */
@Rule(key = "S9134")
public class PydanticFieldValidatorReturnCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Add a return statement to this Pydantic field validator.";

  /**
   * Matches {@code pydantic.functional_validators.field_validator} (direct submodule import):
   *   {@code from pydantic.functional_validators import field_validator}
   * Matches {@code field_validator} re-exported through the {@code pydantic} package root:
   *   {@code from pydantic import field_validator}
   * Uses FQN matching because the pydantic v1 stub does not declare this re-export,
   * so the type resolves to {@code UnresolvedImportType[pydantic.field_validator]}.
   */

  private static final TypeMatcher FIELD_VALIDATOR_MATCHER = TypeMatchers.any(TypeMatchers.isType("pydantic.functional_validators.field_validator"),
    TypeMatchers.withFQN("pydantic.field_validator"));

  private static final TypeMatcher IS_PYDANTIC_MODEL = TypeMatchers.isOrExtendsType("pydantic.BaseModel");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FUNCDEF, PydanticFieldValidatorReturnCheck::checkFunctionDef);
  }

  private static void checkFunctionDef(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();

    if (!hasFieldValidatorDecorator(functionDef, ctx)) {
      return;
    }

    if (!isInsidePydanticModel(functionDef, ctx)) {
      return;
    }

    // Skip stub bodies: functions consisting solely of pass or ellipsis statements
    if (functionDef.body().statements().stream().allMatch(CheckUtils::isEmptyStatement)) {
      return;
    }

    var collector = ReturnCheckUtils.ReturnStmtCollector.collect(functionDef);

    // Generators are not field validators in the conventional sense
    if (collector.containsYield()) {
      return;
    }

    // Use the CFG to detect missing return values on any code path reaching the function exit.
    ControlFlowGraph cfg = ctx.cfg(functionDef);
    if (cfg == null || hasExceptOrFinally(cfg)) {
      // CFG unavailable or contains try/except/finally: fall back to simple presence check.
      // try/except/finally can produce end-predecessor blocks with empty elements() lists,
      // making the CFG traversal unsafe.
      if (collector.getReturnStmts().isEmpty() && !collector.raisesExceptions()) {
        ctx.addIssue(functionDef.name(), MESSAGE);
      }
      return;
    }

    List<Statement> endStatements = cfg.end().predecessors().stream()
      .map(block -> parentStatement(block.elements().get(block.elements().size() - 1)))
      .filter(s -> !s.is(Kind.RAISE_STMT, Kind.ASSERT_STMT, Kind.WITH_STMT) && !isLoopWithReturn(s) && !isIfTruthy(s))
      .toList();

    // If every path reaching the exit ends with a raise/assert, no issue needed.
    if (endStatements.isEmpty()) {
      return;
    }

    // Check whether any path ends without an explicit return <value>
    boolean hasPathWithoutReturnValue = endStatements.stream()
      .anyMatch(s -> !isReturnWithValue(s));

    if (hasPathWithoutReturnValue) {
      ctx.addIssue(functionDef.name(), MESSAGE);
    }
  }

  private static boolean hasFieldValidatorDecorator(FunctionDef functionDef, SubscriptionContext ctx) {
    return functionDef.decorators().stream()
      .anyMatch(d -> FIELD_VALIDATOR_MATCHER.isTrueFor(getDecoratorFunctionExpression(d), ctx));
  }

  private static Expression getDecoratorFunctionExpression(Decorator decorator) {
    Expression expr = decorator.expression();
    if (expr instanceof CallExpression callExpr) {
      return callExpr.callee();
    }
    return expr;
  }

  private static boolean isInsidePydanticModel(FunctionDef functionDef, SubscriptionContext ctx) {
    Tree parent = functionDef.parent();
    if (parent != null) {
      parent = parent.parent();
    }
    if (!(parent instanceof ClassDef classDef)) {
      return false;
    }
    return IS_PYDANTIC_MODEL.isTrueFor(classDef.name(), ctx);
  }

  private static boolean isReturnWithValue(Statement statement) {
    return statement.is(Kind.RETURN_STMT) && !((ReturnStatement) statement).expressions().isEmpty();
  }

  private static boolean hasExceptOrFinally(ControlFlowGraph cfg) {
    return cfg.blocks().stream().anyMatch(block ->
      block instanceof CfgBranchingBlock cfgBranchingBlock
        && cfgBranchingBlock.branchingTree().is(Kind.EXCEPT_CLAUSE, Kind.FINALLY_CLAUSE));
  }

  /**
   * Returns true if the statement is a {@code for} or {@code while} loop whose body contains at
   * least one {@code return} statement (not in a nested function). Such loops cause the CFG to
   * surface the loop statement itself as an implicit exit predecessor (the "loop exhausted" path),
   * which is not a missing-return case when a return exists inside the loop body.
   */
  private static boolean isLoopWithReturn(Statement statement) {
    StatementList body = null;
    if (statement.is(Kind.FOR_STMT)) {
      body = ((ForStatement) statement).body();
    } else if (statement.is(Kind.WHILE_STMT)) {
      body = ((WhileStatement) statement).body();
    }
    if (body == null) {
      return false;
    }
    return bodyContainsReturn(body);
  }

  /**
   * Returns true if the given tree contains a {@code return} statement that is not
   * nested inside a nested function definition or lambda.
   */
  private static boolean bodyContainsReturn(Tree tree) {
    if (tree.is(Kind.RETURN_STMT)) {
      return true;
    }
    if (tree.is(Kind.FUNCDEF, Kind.LAMBDA)) {
      return false;
    }
    return tree.children().stream().anyMatch(PydanticFieldValidatorReturnCheck::bodyContainsReturn);
  }

  /**
   * Returns true if the statement is an {@code if True:} (or equivalent always-truthy condition)
   * with no else branch. Such conditions cause the CFG to surface the {@code IF_STMT} itself as
   * an implicit exit predecessor when the body always returns, which is not a missing-return case.
   */
  private static boolean isIfTruthy(Statement statement) {
    if (!statement.is(Kind.IF_STMT)) {
      return false;
    }
    IfStatement ifStatement = (IfStatement) statement;
    return ifStatement.elseBranch() == null && Expressions.isTruthy(ifStatement.condition());
  }

  private static Statement parentStatement(Tree tree) {
    while (!(tree instanceof Statement)) {
      tree = tree.parent();
    }
    return (Statement) tree;
  }
}
