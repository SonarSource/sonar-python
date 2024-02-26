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
package org.sonar.python.checks.regex;


import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;
import org.sonarsource.analyzer.commons.regex.MatchType;
import org.sonarsource.analyzer.commons.regex.RegexParseResult;
import org.sonarsource.analyzer.commons.regex.finders.RedosFinder;


@Rule(key = "S5852")
public class RedosCheck extends AbstractRegexCheck {

  private static final String MESSAGE = "Make sure the regex used here, which is vulnerable to %s runtime due to backtracking," +
    " cannot lead to denial of service.";
  private static final String EXP = "exponential";
  private static final String POLY = "polynomial";
  private static final Set<String> FULL_MATCH_METHODS = Set.of("fullmatch");
  private static final Set<String> PARTIAL_MATCH_METHODS = Set.of("findall", "search", "split", "sub", "subn");
  private static final String COMPILE_METHOD = "compile";

  @Override
  public void checkRegex(RegexParseResult regexParseResult, CallExpression regexFunctionCall) {
    MatchType matchType = getMatchTypeFromCalledMethod(regexFunctionCall);
    new PythonRedosFinder().checkRegex(regexParseResult, matchType, this::addIssue);
  }

  private static MatchType getMatchTypeFromCalledMethod(CallExpression regexFunctionCall) {
    Symbol symbol = regexFunctionCall.calleeSymbol();
    if (symbol == null) {
      // Defensive, callee symbol should have been checked prior to calling "checkRegex"
      return MatchType.UNKNOWN;
    }
    if (FULL_MATCH_METHODS.contains(symbol.name())) {
      return MatchType.FULL;
    }
    if (PARTIAL_MATCH_METHODS.contains(symbol.name())) {
      return MatchType.PARTIAL;
    }
    if (COMPILE_METHOD.equals(symbol.name())) {
      return matchTypeOfCompiledPattern(regexFunctionCall);
    }
    return MatchType.UNKNOWN;
  }

  private static MatchType matchTypeOfCompiledPattern(CallExpression regexFunctionCall) {
    return Optional.ofNullable(TreeUtils.firstAncestorOfKind(regexFunctionCall, Tree.Kind.ASSIGNMENT_STMT))
      .map(AssignmentStatement.class::cast)
      .map(a -> a.lhsExpressions().get(0).expressions().get(0))
      .filter(lhs -> lhs.is(Tree.Kind.NAME))
      .map(n -> (Name) n)
      .map(HasSymbol::symbol)
      .map(RedosCheck::getMatchTypeFromSymbolUsages)
      .orElse(MatchType.UNKNOWN);
  }

  private static MatchType getMatchTypeFromSymbolUsages(Symbol s) {
    boolean isUsedForFullMatch = s.usages().stream().map(Usage::tree).anyMatch(t -> isUsedInMethod(t, FULL_MATCH_METHODS));
    boolean isUsedForPartialMatch = s.usages().stream().map(Usage::tree).anyMatch(t -> isUsedInMethod(t, PARTIAL_MATCH_METHODS));
    return getMatchType(isUsedForFullMatch, isUsedForPartialMatch);
  }

  private static MatchType getMatchType(boolean isUsedForFullMatch, boolean isUsedForPartialMatch) {
    if (isUsedForFullMatch && isUsedForPartialMatch) {
      return MatchType.BOTH;
    }
    if (isUsedForFullMatch) {
      return MatchType.FULL;
    }
    if (isUsedForPartialMatch) {
      return MatchType.PARTIAL;
    }
    return MatchType.UNKNOWN;
  }

  private static boolean isUsedInMethod(Tree tree, Set<String> methodNames) {
    return TreeUtils.firstAncestor(tree, isCallToMethod(methodNames)) != null;
  }

  private static Predicate<Tree> isCallToMethod(Set<String> methodNames) {
    return tree -> Optional.ofNullable(tree)
      .filter(t -> t.is(Tree.Kind.CALL_EXPR))
      .map(CallExpression.class::cast)
      .map(CallExpression::calleeSymbol)
      .map(Symbol::name)
      .filter(methodNames::contains)
      .isPresent();
  }

  static class PythonRedosFinder extends RedosFinder {

    @Override
    protected Optional<String> message(RedosFinder.BacktrackingType backtrackingType, boolean regexContainsBackReference) {
      switch (backtrackingType) {
        case ALWAYS_EXPONENTIAL:
        case QUADRATIC_WHEN_OPTIMIZED:
        case LINEAR_WHEN_OPTIMIZED:
          return Optional.of(String.format(MESSAGE, EXP));
        case ALWAYS_QUADRATIC:
          return Optional.of(String.format(MESSAGE, POLY));
        default:
          return Optional.empty();
      }
    }
  }
}
