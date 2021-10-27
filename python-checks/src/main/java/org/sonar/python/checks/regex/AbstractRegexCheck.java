/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.python.checks.regex;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.regex.RegexContext;
import org.sonar.python.tree.TreeUtils;
import org.sonarsource.analyzer.commons.regex.RegexIssueLocation;
import org.sonarsource.analyzer.commons.regex.RegexParseResult;
import org.sonarsource.analyzer.commons.regex.ast.RegexSyntaxElement;

public abstract class AbstractRegexCheck extends PythonSubscriptionCheck {

  private static final Set<String> REGEX_FUNCTIONS = new HashSet<>(Arrays.asList("re.sub", "re.subn", "re.compile", "re.search", "re.match",
    "re.fullmatch", "re.split", "re.findall", "re.finditer"));
  protected RegexContext regexContext;

  // We want to report only one issue per element for one rule.
  final Set<RegexSyntaxElement> reportedRegexTrees = new HashSet<>();

  protected Set<String> lookedUpFunctionNames() {
    return REGEX_FUNCTIONS;
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkCall);
  }

  public abstract void checkRegex(RegexParseResult regexParseResult, CallExpression regexFunctionCall);

  private void checkCall(SubscriptionContext ctx) {
    regexContext = (RegexContext) ctx;
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Symbol calleeSymbol = callExpression.calleeSymbol();
    if (calleeSymbol == null || calleeSymbol.fullyQualifiedName() == null) {
      return;
    }
    if (lookedUpFunctionNames().contains(calleeSymbol.fullyQualifiedName())) {
      patternArgStringLiteral(callExpression)
        .flatMap(this::regexForStringLiteral)
        .ifPresent(parseResult -> checkRegex(parseResult, callExpression));
    }
  }

  private Optional<RegexParseResult> regexForStringLiteral(StringLiteral literal) {
    // TODO: for now we only handle strings with an "r" prefix. This will be extended.
    if (literal.stringElements().size() == 1 && "r".equalsIgnoreCase(literal.stringElements().get(0).prefix())) {
      return Optional.of(regexContext.regexForStringElement(literal.stringElements().get(0)));
    }
    return Optional.empty();
  }

  private static Optional<StringLiteral> patternArgStringLiteral(CallExpression regexFunctionCall) {
    RegularArgument patternArgument = TreeUtils.nthArgumentOrKeyword(0, "pattern", regexFunctionCall.arguments());
    if (patternArgument == null) {
      return Optional.empty();
    }
    Expression patternArgumentExpression = patternArgument.expression();
    if (patternArgumentExpression.is(Tree.Kind.STRING_LITERAL)) {
      return Optional.of((StringLiteral) patternArgumentExpression);
    }
    return Optional.empty();
  }

  public void addIssue(RegexSyntaxElement regexTree, String message, @Nullable Integer cost, List<RegexIssueLocation> secondaries) {
    if (reportedRegexTrees.add(regexTree)) {
      regexContext.addIssue(regexTree, message);
      // TODO: Add secondary location to the issue SONARPY-886
      // TODO: Add cost to the issue SONARPY-893
    }
  }



}
