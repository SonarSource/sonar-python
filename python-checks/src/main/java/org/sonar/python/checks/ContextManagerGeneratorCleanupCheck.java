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
import java.util.Set;
import java.util.Objects;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.CaseBlock;
import org.sonar.plugins.python.api.tree.CompoundAssignmentStatement;
import org.sonar.plugins.python.api.tree.ElseClause;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.MatchStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TryStatement;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.tree.WhileStatement;
import org.sonar.plugins.python.api.tree.WithItem;
import org.sonar.plugins.python.api.tree.WithStatement;
import org.sonar.plugins.python.api.tree.YieldExpression;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.checks.utils.UnittestUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S9152")
public class ContextManagerGeneratorCleanupCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Cleanup after this \"yield\" may be skipped on early exit.";
  private static final String SECONDARY_MESSAGE = "Move this into \"try\"/\"finally\".";
  private static final String SWALLOWED_MESSAGE = "Swallowing this exception may skip cleanup of other context managers.";
  private static final String RERAISE_MESSAGE = "Re-raise this exception.";

  private static final Tree.Kind[] JUMPS = {Tree.Kind.PASS_STMT, Tree.Kind.BREAK_STMT, Tree.Kind.CONTINUE_STMT};
  private static final Tree.Kind[] ASSIGNMENTS = {Tree.Kind.ASSIGNMENT_STMT, Tree.Kind.ANNOTATED_ASSIGNMENT, Tree.Kind.COMPOUND_ASSIGNMENT};
  private static final Tree.Kind[] SELECTIONS = {Tree.Kind.IF_STMT, Tree.Kind.MATCH_STMT};
  private static final Tree.Kind[] LOOPS = {Tree.Kind.FOR_STMT, Tree.Kind.WHILE_STMT};

  private static final TypeMatcher CONTEXT_MANAGER = TypeMatchers.any(
    TypeMatchers.isType("contextlib.contextmanager"),
    TypeMatchers.isType("contextlib.asynccontextmanager"));

  private static final TypeMatcher BASE_EXCEPTION = TypeMatchers.isType("BaseException");

  private static final TypeMatcher REPORTING_CALL = TypeMatchers.any(Stream.concat(
    Stream.of(
      TypeMatchers.isType("builtins.print"),
      TypeMatchers.isType("warnings.warn"),
      TypeMatchers.isType("warnings.warn_explicit"),
      TypeMatchers.isType("unittest.case.TestCase.fail"),
      TypeMatchers.isType("pytest.fail"),
      TypeMatchers.withFQN("pytest.fail")),
    Stream.of("logging", "logging.Logger", "logging.LoggerAdapter")
      .flatMap(owner -> Stream.of("debug", "info", "warning", "warn", "error", "exception", "critical", "log")
        .map(method -> TypeMatchers.isType(owner + "." + method)))));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ContextManagerGeneratorCleanupCheck::checkFunctionDef);
  }

  private static void checkFunctionDef(SubscriptionContext ctx) {
    if (ctx.isLikelyTestFile()) {
      return;
    }
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
    if (!isContextManager(functionDef, ctx)) {
      return;
    }
    YieldCollector collector = new YieldCollector();
    functionDef.body().accept(collector);
    for (YieldExpression yieldExpression : collector.found) {
      Statement enclosing = TreeUtils.firstAncestorOfClass(yieldExpression, Statement.class);
      if (enclosing == null) {
        continue;
      }
      List<Statement> cleanup = unprotectedCleanupAfter(enclosing, ctx);
      if (!cleanup.isEmpty()) {
        PreciseIssue issue = ctx.addIssue(yieldExpression, MESSAGE);
        cleanup.forEach(statement -> issue.secondary(statement, SECONDARY_MESSAGE));
      }
    }
    checkSwallowedBaseException(collector.found, ctx);
  }

  /**
   * A catch-all handler that does not re-raise swallows the error thrown back in at {@code yield}.
   * Enclosing context managers then exit normally and skip the cleanup they run on failure.
   */
  private static void checkSwallowedBaseException(List<YieldExpression> yields, SubscriptionContext ctx) {
    Set<ExceptClause> reported = new HashSet<>();
    yields.forEach(yieldExpression -> reportSwallowedAround(yieldExpression, reported, ctx));
  }

  /**
   * Reports a catch-all handler around {@code yieldExpression} that hides an exception from an enclosing context manager.
   * Without an enclosing manager in this function, suppression or deferred propagation may be the generator's contract;
   * the rule cannot infer the cleanup policy of callers.
   * A {@code finally} still runs after that handler, so it is left to re-raise or clean up.
   */
  private static void reportSwallowedAround(YieldExpression yieldExpression, Set<ExceptClause> reported, SubscriptionContext ctx) {
    Tree current = yieldExpression.parent();
    while (current != null && !current.is(Tree.Kind.FUNCDEF)) {
      if (current instanceof TryStatement tryStatement && tryStatement.finallyClause() == null
        && isInside(yieldExpression, tryStatement.body()) && hasEnclosingContextManager(tryStatement)) {
        tryStatement.exceptClauses().forEach(exceptClause -> reportSwallowed(exceptClause, reported, ctx));
      }
      current = current.parent();
    }
  }

  private static boolean hasEnclosingContextManager(TryStatement tryStatement) {
    return TreeUtils.firstAncestorOfKind(tryStatement, Tree.Kind.WITH_STMT, Tree.Kind.FUNCDEF) instanceof WithStatement;
  }

  private static void reportSwallowed(ExceptClause exceptClause, Set<ExceptClause> reported, SubscriptionContext ctx) {
    if (!catchesBaseException(exceptClause, ctx) || endsWithRaise(exceptClause) || !reported.add(exceptClause)) {
      return;
    }
    List<Statement> statements = exceptClause.body().statements();
    PreciseIssue issue = ctx.addIssue(exceptClause.exceptKeyword(), SWALLOWED_MESSAGE);
    if (!statements.isEmpty()) {
      issue.secondary(statements.get(statements.size() - 1), RERAISE_MESSAGE);
    }
  }

  private static boolean isInside(Tree node, Tree ancestor) {
    Tree current = node;
    while (current != null && current != ancestor) {
      current = current.parent();
    }
    return current == ancestor;
  }

  private static boolean isContextManager(FunctionDef functionDef, SubscriptionContext ctx) {
    return functionDef.decorators().stream().anyMatch(decorator -> {
      Expression expression = decorator.expression();
      if (expression instanceof CallExpression callExpression) {
        expression = callExpression.callee();
      }
      return CONTEXT_MANAGER.isTrueFor(expression, ctx);
    });
  }

  /**
   * Collects the cleanup that runs after {@code yield} only when the generator is resumed, by walking outward from the
   * statement holding the yield up to the function body.
   */
  private static List<Statement> unprotectedCleanupAfter(Statement yieldStatement, SubscriptionContext ctx) {
    List<Statement> cleanup = new ArrayList<>();
    Tree current = yieldStatement;
    while (current != null && !current.is(Tree.Kind.FUNCDEF, Tree.Kind.FILE_INPUT)) {
      if (current.parent() instanceof StatementList block) {
        cleanup.addAll(cleanupAfter(block, current, ctx));
        if (statementsAfter(block, current).stream().anyMatch(statement -> statement.is(Tree.Kind.RETURN_STMT))) {
          break;
        }
        current = block.parent();
      } else {
        current = current.parent();
      }
    }
    return cleanup;
  }

  /**
   * Picks out the statements of one block that run after {@code current} and that nothing else already protects.
   */
  private static List<Statement> cleanupAfter(StatementList block, Tree current, SubscriptionContext ctx) {
    if (runsOnSuccessPathOnly(block, ctx)) {
      return List.of();
    }
    // A loop right after a protected try consumes what that try produced, so it is result handling rather than cleanup.
    boolean protectedLoops = current instanceof TryStatement tryStatement && tryStatement.finallyClause() != null;
    return statementsAfter(block, current).stream()
      .takeWhile(statement -> !statement.is(Tree.Kind.RETURN_STMT))
      .filter(statement -> !(protectedLoops && statement.is(LOOPS)))
      .filter(statement -> isUnprotectedCleanup(statement, ctx))
      .toList();
  }

  private static List<Statement> statementsAfter(StatementList block, Tree current) {
    List<Statement> statements = block.statements();
    int index = statements.indexOf(current);
    return index < 0 ? List.of() : statements.subList(index + 1, statements.size());
  }

  /**
   * True when the enclosing construct already releases the resource, which makes trailing statements success-path work.
   * A try whose body holds the yield is success-path-only when a {@code finally} runs, or when every {@code except}
   * re-raises and either the enclosing {@code with} binds a name used in that body, or a catch-all {@code except}
   * cleans up before it re-raises. A handler that only logs, or an unrelated {@code with}, does not release anything.
   */
  private static boolean runsOnSuccessPathOnly(StatementList block, SubscriptionContext ctx) {
    if (block.parent() instanceof TryStatement tryStatement && tryStatement.body() == block) {
      return tryStatement.finallyClause() != null
        || (alwaysReraises(tryStatement) && (withTargetUsedIn(tryStatement) || catchAllReleases(tryStatement, ctx)));
    }
    return block.parent() instanceof WithStatement;
  }

  /**
   * True when a {@code with} between this try and the enclosing function binds a name that the try body uses.
   * A {@code with} that does not name that resource, such as {@code with lock:}, does not count.
   */
  private static boolean withTargetUsedIn(TryStatement tryStatement) {
    Tree ancestor = TreeUtils.firstAncestorOfKind(tryStatement, Tree.Kind.WITH_STMT, Tree.Kind.FUNCDEF);
    if (!(ancestor instanceof WithStatement withStatement)) {
      return false;
    }
    return withStatement.withItems().stream()
      .map(WithItem::expression)
      .filter(Name.class::isInstance)
      .map(Name.class::cast)
      .map(Name::symbolV2)
      .filter(Objects::nonNull)
      .anyMatch(symbol -> usesSymbol(tryStatement.body(), symbol));
  }

  private static boolean usesSymbol(Tree tree, SymbolV2 symbol) {
    return TreeUtils.hasDescendant(tree, descendant -> descendant instanceof Name name && symbol.equals(name.symbolV2()));
  }

  /**
   * True when a bare {@code except} or an {@code except BaseException} does real cleanup before re-raising.
   * Logging or a bare {@code raise} does not release the resource.
   */
  private static boolean catchAllReleases(TryStatement tryStatement, SubscriptionContext ctx) {
    return tryStatement.exceptClauses().stream().anyMatch(clause -> catchesBaseException(clause, ctx) && releasesBeforeRaise(clause, ctx));
  }

  private static boolean catchesBaseException(ExceptClause exceptClause, SubscriptionContext ctx) {
    Expression exception = exceptClause.exception();
    if (exception == null) {
      return true;
    }
    if (exception instanceof Tuple tuple) {
      return tuple.elements().stream().anyMatch(element -> BASE_EXCEPTION.isTrueFor(element, ctx));
    }
    return BASE_EXCEPTION.isTrueFor(exception, ctx);
  }

  /**
   * True when every path that re-raises has already performed cleanup, including cleanup nested in the {@code with},
   * {@code if}, or {@code try} that holds the {@code raise}.
   */
  private static boolean releasesBeforeRaise(ExceptClause exceptClause, SubscriptionContext ctx) {
    return releasesBeforeRaise(exceptClause.body().statements(), ctx);
  }

  private static boolean releasesBeforeRaise(List<Statement> statements, SubscriptionContext ctx) {
    if (!alwaysRaises(statements)) {
      return false;
    }
    boolean cleanedBefore = statements.subList(0, statements.size() - 1).stream()
      .anyMatch(statement -> roleOf(statement, ctx) == Role.CLEANUP);
    return cleanedBefore || releasesInside(statements.get(statements.size() - 1), ctx);
  }

  private static boolean releasesInside(Statement statement, SubscriptionContext ctx) {
    if (statement instanceof WithStatement withStatement) {
      return releasesBeforeRaise(withStatement.statements().statements(), ctx);
    }
    if (statement instanceof IfStatement ifStatement) {
      return ifReleases(ifStatement, ctx);
    }
    if (statement instanceof TryStatement tryStatement) {
      return tryReleases(tryStatement, ctx);
    }
    return false;
  }

  private static boolean ifReleases(IfStatement ifStatement, SubscriptionContext ctx) {
    ElseClause elseBranch = ifStatement.elseBranch();
    if (!releasesBeforeRaise(ifStatement.body().statements(), ctx) || !releasesBeforeRaise(elseBranch.body().statements(), ctx)) {
      return false;
    }
    return ifStatement.elifBranches().stream().allMatch(elif -> releasesBeforeRaise(elif.body().statements(), ctx));
  }

  private static boolean tryReleases(TryStatement tryStatement, SubscriptionContext ctx) {
    var finallyClause = tryStatement.finallyClause();
    if (finallyClause != null && alwaysRaises(finallyClause.body().statements())) {
      return hasCleanup(tryStatement.body().statements(), ctx)
        || releasesBeforeRaise(finallyClause.body().statements(), ctx);
    }
    return releasesBeforeRaise(tryStatement.body().statements(), ctx)
      && tryStatement.exceptClauses().stream().allMatch(clause -> releasesBeforeRaise(clause.body().statements(), ctx));
  }

  private static boolean hasCleanup(List<Statement> statements, SubscriptionContext ctx) {
    return statements.stream().anyMatch(statement -> roleOf(statement, ctx) == Role.CLEANUP);
  }

  /**
   * True when the try has at least one {@code except} clause and every one of them ends with a {@code raise},
   * guaranteeing that any exception keeps propagating instead of being swallowed.
   */
  private static boolean alwaysReraises(TryStatement tryStatement) {
    List<ExceptClause> exceptClauses = tryStatement.exceptClauses();
    return !exceptClauses.isEmpty() && exceptClauses.stream().allMatch(ContextManagerGeneratorCleanupCheck::endsWithRaise);
  }

  private static boolean endsWithRaise(ExceptClause exceptClause) {
    return alwaysRaises(exceptClause.body().statements());
  }

  /**
   * True when every way through these statements ends in {@code raise}, including a raise nested in {@code if},
   * {@code with}, or {@code try}.
   */
  private static boolean alwaysRaises(List<Statement> statements) {
    return !statements.isEmpty() && alwaysRaises(statements.get(statements.size() - 1));
  }

  private static boolean alwaysRaises(Statement statement) {
    if (statement.is(Tree.Kind.RAISE_STMT)) {
      return true;
    }
    if (statement instanceof IfStatement ifStatement) {
      return ifAlwaysRaises(ifStatement);
    }
    if (statement instanceof WithStatement withStatement) {
      return alwaysRaises(withStatement.statements().statements());
    }
    if (statement instanceof TryStatement tryStatement) {
      return tryAlwaysRaises(tryStatement);
    }
    return false;
  }

  private static boolean ifAlwaysRaises(IfStatement ifStatement) {
    ElseClause elseBranch = ifStatement.elseBranch();
    if (elseBranch == null || !alwaysRaises(ifStatement.body().statements()) || !alwaysRaises(elseBranch.body().statements())) {
      return false;
    }
    return ifStatement.elifBranches().stream().allMatch(elif -> alwaysRaises(elif.body().statements()));
  }

  private static boolean tryAlwaysRaises(TryStatement tryStatement) {
    var finallyClause = tryStatement.finallyClause();
    if (finallyClause != null && alwaysRaises(finallyClause.body().statements())) {
      return true;
    }
    return alwaysRaises(tryStatement.body().statements())
      && tryStatement.exceptClauses().stream().allMatch(ContextManagerGeneratorCleanupCheck::endsWithRaise);
  }

  /**
   * True when a statement left after the yield is cleanup the caller may never reach. A bare jump releases nothing, and
   * a statement that only observes state loses nothing by being skipped, but a plain assignment does restore state.
   */
  private static boolean isUnprotectedCleanup(Statement statement, SubscriptionContext ctx) {
    return !isJump(statement) && roleOf(statement, ctx) != Role.OBSERVATION;
  }

  /**
   * True for a bare {@code pass}, {@code break} or {@code continue}, and for an {@code if} or {@code match} whose
   * branches contain nothing else, since whichever branch runs nothing is released. Loops are not included: driving
   * the iterable is work even when the body does nothing, so {@code for x in xs: pass} stays cleanup.
   */
  private static boolean isJump(Statement statement) {
    if (statement.is(JUMPS)) {
      return true;
    }
    return statement.is(SELECTIONS)
      && branches(statement).stream().flatMap(branch -> branch.statements().stream())
        .allMatch(ContextManagerGeneratorCleanupCheck::isJump);
  }

  /**
   * Classifies a statement, recursing into the branches of a selection or a loop: a branch that releases state makes
   * the whole statement cleanup, otherwise a single observation anywhere makes it an observation.
   *
   * @see Role
   */
  private static Role roleOf(Statement statement, SubscriptionContext ctx) {
    if (statement.is(JUMPS) || statement.is(ASSIGNMENTS)) {
      return Role.NEUTRAL;
    }
    if (isObservation(statement, ctx)) {
      return Role.OBSERVATION;
    }
    List<StatementList> branches = branches(statement);
    return branches.isEmpty() ? Role.CLEANUP : roleOfBranches(branches, ctx);
  }

  /**
   * Combines the roles of everything the branches of a selection or a loop contain.
   */
  private static Role roleOfBranches(List<StatementList> branches, SubscriptionContext ctx) {
    List<Role> innerRoles = branches.stream()
      .flatMap(branch -> branch.statements().stream())
      .map(inner -> roleOf(inner, ctx))
      .toList();
    if (innerRoles.contains(Role.CLEANUP) || branches.stream().anyMatch(ContextManagerGeneratorCleanupCheck::restoresState)) {
      return Role.CLEANUP;
    }
    return innerRoles.contains(Role.OBSERVATION) ? Role.OBSERVATION : Role.NEUTRAL;
  }

  /**
   * True when a branch does nothing but assign, and at least one of those assignments writes state that outlives the
   * generator. Writing only locals prepares a value for a following statement, such as an assertion message.
   */
  private static boolean restoresState(StatementList branch) {
    List<Statement> statements = branch.statements();
    return statements.stream().allMatch(statement -> statement.is(ASSIGNMENTS))
      && statements.stream().flatMap(ContextManagerGeneratorCleanupCheck::assignedTargets)
        .anyMatch(target -> target.is(Tree.Kind.QUALIFIED_EXPR, Tree.Kind.SUBSCRIPTION));
  }

  private static Stream<Expression> assignedTargets(Statement statement) {
    if (statement instanceof AssignmentStatement assignment) {
      return assignment.lhsExpressions().stream().flatMap(targets -> targets.expressions().stream());
    }
    if (statement instanceof AnnotatedAssignment annotated) {
      return Stream.of(annotated.variable());
    }
    if (statement instanceof CompoundAssignmentStatement compound) {
      return Stream.of(compound.lhsExpression());
    }
    return Stream.empty();
  }

  /**
   * True for a statement that reports or checks state. Raising is included whatever the exception: it signals a
   * problem rather than releasing a resource, and there is no meaningful way to move it into a {@code finally}.
   */
  private static boolean isObservation(Statement statement, SubscriptionContext ctx) {
    if (statement.is(Tree.Kind.ASSERT_STMT, Tree.Kind.RAISE_STMT)) {
      return true;
    }
    return statement instanceof ExpressionStatement expressionStatement
      && expressionStatement.expressions().size() == 1
      && expressionStatement.expressions().get(0) instanceof CallExpression call
      && isReportingCall(call.callee(), ctx);
  }

  private static boolean isReportingCall(Expression callee, SubscriptionContext ctx) {
    return UnittestUtils.isUnittestAssertion(callee, ctx)
      || REPORTING_CALL.isTrueFor(callee, ctx)
      // Third-party TestCase subclasses often lack stubs, so fall back on the self.fail shape.
      || (callee instanceof QualifiedExpression qualified && CheckUtils.isSelf(qualified.qualifier()) && "fail".equals(qualified.name().name()));
  }

  /**
   * The bodies a statement may run conditionally, or an empty list for a statement whose contents are not branches.
   */
  private static List<StatementList> branches(Statement statement) {
    if (statement instanceof IfStatement ifStatement) {
      List<StatementList> branches = new ArrayList<>();
      branches.add(ifStatement.body());
      ifStatement.elifBranches().forEach(elif -> branches.add(elif.body()));
      return withElse(branches, ifStatement.elseBranch());
    }
    if (statement instanceof MatchStatement matchStatement) {
      return matchStatement.caseBlocks().stream().map(CaseBlock::body).toList();
    }
    if (statement instanceof ForStatement forStatement) {
      return withElse(List.of(forStatement.body()), forStatement.elseClause());
    }
    if (statement instanceof WhileStatement whileStatement) {
      return withElse(List.of(whileStatement.body()), whileStatement.elseClause());
    }
    return List.of();
  }

  private static List<StatementList> withElse(List<StatementList> bodies, @Nullable ElseClause elseClause) {
    if (elseClause == null) {
      return bodies;
    }
    List<StatementList> branches = new ArrayList<>(bodies);
    branches.add(elseClause.body());
    return branches;
  }

  /**
   * What a statement after the yield does: release a resource, report or check state, or neither.
   */
  private enum Role {
    CLEANUP, OBSERVATION, NEUTRAL
  }

  private static final class YieldCollector extends BaseTreeVisitor {
    private final List<YieldExpression> found = new ArrayList<>();

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      // Nested functions are visited on their own FUNCDEF event.
    }

    @Override
    public void visitLambda(LambdaExpression lambdaExpression) {
      // Lambdas cannot yield.
    }

    @Override
    public void visitYieldExpression(YieldExpression yieldExpression) {
      found.add(yieldExpression);
    }
  }
}
