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

import java.util.Comparator;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.YieldExpression;
import org.sonar.plugins.python.api.tree.YieldStatement;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S9407")
public class YieldFromDelegationCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Replace this loop with a \"yield from\" statement.";
  private static final String SECONDARY_MESSAGE = "This \"yield\" only relays the loop variable.";
  private static final String QUICK_FIX_MESSAGE = "Replace this loop with \"yield from\"";

  // Ordering and the before/after helpers share one definition so that they cannot disagree:
  // the comparator uses pythonLine/pythonColumn, which differ from line/column for compressed tokens.
  private static final Comparator<Tree> BY_POSITION = TreeUtils.getTreeByPositionComparator();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FOR_STMT, YieldFromDelegationCheck::check);
  }

  private static void check(SubscriptionContext ctx) {
    ForStatement forStatement = (ForStatement) ctx.syntaxNode();
    // "yield from" is a syntax error in an async generator, and an "async for" cannot be delegated to at all.
    if (forStatement.isAsync() || TreeUtils.asyncTokenOfEnclosingFunction(forStatement).isPresent()) {
      return;
    }
    // The else clause runs only when the loop completes normally, which "yield from" cannot express.
    if (forStatement.elseClause() != null) {
      return;
    }
    // "for value, in rows" unpacks each row into a one-element tuple, it does not iterate over the rows themselves.
    Name loopVariable = singleName(forStatement.expressions());
    if (loopVariable == null || hasTargetComma(forStatement, loopVariable)) {
      return;
    }
    YieldStatement yieldStatement = onlyYieldStatement(forStatement.body().statements());
    if (yieldStatement == null) {
      return;
    }
    YieldExpression yieldExpression = yieldStatement.yieldExpression();
    if (yieldExpression.fromKeyword() != null) {
      return;
    }
    // "yield value," yields a one-element tuple, not the element itself.
    Name yieldedName = singleName(yieldExpression.expressions());
    if (yieldedName == null || hasYieldComma(ctx, yieldedName)) {
      return;
    }
    if (!isSameSymbol(loopVariable, yieldedName) || isReadAfterLoop(loopVariable, forStatement)) {
      return;
    }
    PreciseIssue issue = ctx.addIssue(forStatement.forKeyword(), MESSAGE).secondary(yieldStatement, SECONDARY_MESSAGE);
    addQuickFix(issue, forStatement, yieldExpression);
  }

  private static void addQuickFix(PreciseIssue issue, ForStatement forStatement, YieldExpression yieldExpression) {
    List<Expression> testExpressions = forStatement.testExpressions();
    // "for x in a, b" iterates over an implicit tuple, which "yield from a, b" would not reproduce.
    if (testExpressions.size() != 1 || hasCommentToPreserve(forStatement, yieldExpression)) {
      return;
    }
    // A multi-line iterable cannot be rendered back to source reliably, so no fix is offered for it.
    String iterable = TreeUtils.treeToString(testExpressions.get(0), false);
    if (iterable == null) {
      return;
    }
    // The range stops at the yielded name so that the newline, and any comment trailing the yield, are kept.
    PythonTextEdit edit = TextEditUtils.replaceRange(forStatement.forKeyword(), yieldExpression, "yield from " + iterable);
    issue.addQuickFix(PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE, edit));
  }

  // The replaced range covers the loop header and the yield, so a comment inside it would be dropped by the fix.
  // A comment preceding the loop, or trailing the yield, sits outside that range and is left alone.
  private static boolean hasCommentToPreserve(ForStatement forStatement, YieldExpression yieldExpression) {
    return CheckUtils.hasCommentBetween(forStatement, forStatement.forKeyword(), yieldExpression.lastToken());
  }

  /**
   * "for value, in rows" unpacks each row into a one-element tuple. The loop target commas are kept as children of
   * the for statement, so the comma following the target can be found in the tree.
   */
  private static boolean hasTargetComma(ForStatement forStatement, Expression target) {
    return forStatement.children().stream()
      .filter(child -> child.is(Tree.Kind.TOKEN))
      .map(Token.class::cast)
      .anyMatch(token -> ",".equals(token.value()) && isAfter(token, target) && isBeforeOrAt(token, forStatement.inKeyword()));
  }

  /**
   * "yield value," yields a one-element tuple. Unlike the loop target commas, {@code YieldExpressionImpl} does not
   * keep its commas as children, so this one is not reachable from the tree and the source line is read instead.
   * The lookup uses pythonLine/pythonColumn because those, unlike line/column, are positions inside the content
   * returned by {@code pythonFile().content()} for a notebook.
   */
  private static boolean hasYieldComma(SubscriptionContext ctx, Expression yielded) {
    Token last = yielded.lastToken();
    String line = sourceLine(ctx.pythonFile().content(), last.pythonLine().line());
    int index = last.pythonColumn() + last.value().length();
    while (index < line.length() && Character.isWhitespace(line.charAt(index))) {
      index++;
    }
    return index < line.length() && line.charAt(index) == ',';
  }

  // Only the one line holding the yield is needed, so the file is walked to it instead of being split into an array.
  private static String sourceLine(String content, int line) {
    int start = 0;
    for (int remaining = line - 1; remaining > 0; remaining--) {
      start = endOfLine(content, start);
      if (start >= content.length()) {
        return "";
      }
      start += content.startsWith("\r\n", start) ? 2 : 1;
    }
    return content.substring(start, endOfLine(content, start));
  }

  private static int endOfLine(String content, int from) {
    for (int index = from; index < content.length(); index++) {
      char character = content.charAt(index);
      if (character == '\n' || character == '\r') {
        return index;
      }
    }
    return content.length();
  }

  @CheckForNull
  private static Name singleName(List<Expression> expressions) {
    if (expressions.size() != 1) {
      return null;
    }
    return TreeUtils.toOptionalInstanceOf(Name.class, expressions.get(0)).orElse(null);
  }

  @CheckForNull
  private static YieldStatement onlyYieldStatement(List<Statement> statements) {
    if (statements.size() != 1) {
      return null;
    }
    return TreeUtils.toOptionalInstanceOf(YieldStatement.class, statements.get(0)).orElse(null);
  }

  private static boolean isSameSymbol(Name loopVariable, Name yieldedName) {
    SymbolV2 loopSymbol = loopVariable.symbolV2();
    return loopSymbol != null && loopSymbol == yieldedName.symbolV2();
  }

  // Usages that give the name a new value without reading the previous one. Listed explicitly rather than relying on
  // UsageV2#isBindingUsage, which also covers "value += 1" and "nonlocal value", neither of which rebinds on its own.
  private static final Set<UsageV2.Kind> REBINDING_KINDS = EnumSet.of(
    UsageV2.Kind.ASSIGNMENT_LHS,
    UsageV2.Kind.LOOP_DECLARATION,
    UsageV2.Kind.COMP_DECLARATION,
    UsageV2.Kind.PARAMETER,
    UsageV2.Kind.IMPORT,
    UsageV2.Kind.FUNC_DECLARATION,
    UsageV2.Kind.CLASS_DECLARATION,
    UsageV2.Kind.EXCEPTION_INSTANCE,
    UsageV2.Kind.WITH_INSTANCE,
    UsageV2.Kind.PATTERN_DECLARATION,
    UsageV2.Kind.TYPE_PARAM_DECLARATION,
    UsageV2.Kind.TYPE_ALIAS_DECLARATION);

  // After the rewrite the loop variable no longer exists, so reading it once the loop is over would break.
  // Only the first statement following the loop matters: a rebinding gives the name a value before any later read.
  // The whole statement is considered, because "value = transform(value)" both rebinds the name and reads it.
  private static boolean isReadAfterLoop(Name loopVariable, ForStatement forStatement) {
    SymbolV2 symbol = loopVariable.symbolV2();
    if (symbol == null) {
      return false;
    }
    Token loopEnd = forStatement.lastToken();
    List<UsageV2> usagesAfterLoop = symbol.usages().stream()
      .filter(usage -> isAfter(usage.tree(), loopEnd))
      .sorted(Comparator.comparing(UsageV2::tree, BY_POSITION))
      .toList();
    if (usagesAfterLoop.isEmpty()) {
      return false;
    }
    Tree firstStatement = enclosingStatement(usagesAfterLoop.get(0).tree());
    return firstStatement == null || usagesAfterLoop.stream()
      .filter(usage -> firstStatement.equals(enclosingStatement(usage.tree())))
      .anyMatch(usage -> !isRebinding(usage));
  }

  @CheckForNull
  private static Tree enclosingStatement(Tree tree) {
    return TreeUtils.firstAncestor(tree, ancestor -> ancestor.parent() != null && ancestor.parent().is(Tree.Kind.STATEMENT_LIST));
  }

  private static boolean isRebinding(UsageV2 usage) {
    return REBINDING_KINDS.contains(usage.kind()) && !isAnnotationWithoutValue(usage.tree());
  }

  // "value: int" is reported as a binding usage but assigns nothing, so the name stays undefined at runtime.
  private static boolean isAnnotationWithoutValue(Tree tree) {
    return TreeUtils.toOptionalInstanceOf(AnnotatedAssignment.class, tree.parent())
      .filter(annotatedAssignment -> annotatedAssignment.assignedValue() == null)
      .isPresent();
  }

  private static boolean isAfter(Tree tree, Tree reference) {
    return tree.firstToken() != null && BY_POSITION.compare(tree, reference) > 0;
  }

  private static boolean isBeforeOrAt(Tree tree, Tree reference) {
    return !isAfter(tree, reference);
  }
}
