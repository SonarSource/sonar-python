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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonVisitorCheck;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S1192")
public class StringLiteralDuplicationCheck extends PythonVisitorCheck {

  private static final Integer MINIMUM_LITERAL_LENGTH = 5;
  private static final int DEFAULT_THRESHOLD = 3;
  private static final String DEFAULT_EXCLUSION_REGEX = "[_\\-a-zA-Z0-9]+"
  private static final Pattern FORMATTING_PATTERN = Pattern.compile("[0-9{} .-_%:dfrsymhYMHS]+");
  private static final Pattern COLOR_PATTERN = Pattern.compile("#[0-9a-fA-F]{6}");

  @RuleProperty(
    key = "threshold",
    description = "Number of times a literal must be duplicated to trigger an issue",
    defaultValue = "" + DEFAULT_THRESHOLD)
  public int threshold = DEFAULT_THRESHOLD;

  @RuleProperty(
    key = "exclusionRegex",
    description = "RegEx matching literals to exclude from triggering an issue",
    defaultValue = DEFAULT_EXCLUSION_REGEX)
  public String exclusionRegex = DEFAULT_EXCLUSION_REGEX;

  private Map<String, List<StringLiteral>> literalsByValue = new HashMap<>();

  @Override
  public void visitFileInput(FileInput fileInput) {
    literalsByValue.clear();

    super.visitFileInput(fileInput);

    for (Map.Entry<String, List<StringLiteral>> entry : literalsByValue.entrySet()) {
      List<StringLiteral> occurrences = entry.getValue();
      int nbOfOccurrences = occurrences.size();
      if (nbOfOccurrences >= threshold) {
        StringLiteral first = occurrences.get(0);
        String message = String.format(
          "Define a constant instead of duplicating this literal %s %s times.",
          first.firstToken().value(),
          nbOfOccurrences);
        PreciseIssue issue = addIssue(first, message).withCost(nbOfOccurrences - 1);
        occurrences.stream()
          .skip(1)
          .forEach(stringLiteral -> issue.secondary(stringLiteral, "Duplication"));
      }
    }
  }

  @Override
  public void visitExpressionStatement(ExpressionStatement expressionStatement) {
    // exclude docstrings
    if (!expressionStatement.expressions().get(0).is(Tree.Kind.STRING_LITERAL)) {
      super.visitExpressionStatement(expressionStatement);
    }
  }

  @Override
  public void visitStringLiteral(StringLiteral literal) {
    String value = Expressions.unescape(literal);
    boolean hasInterpolation = literal.stringElements().stream().anyMatch(StringElement::isInterpolated);
    try  {
      Pattern exclusionPattern = Pattern.compile(exclusionRegex);
    } catch (PatternSyntaxException e){
      throw new IllegalStateException("Unable to compile regular expression: " + exclusionRegex, e);
    }
    boolean isExcluded = hasInterpolation
      || value.length() < MINIMUM_LITERAL_LENGTH
      || exclusionPattern.matcher(value).matches()
      || FORMATTING_PATTERN.matcher(value).matches()
      || COLOR_PATTERN.matcher(value).matches();
    if (!isExcluded) {
      String valueWithQuotes = TreeUtils.tokens(literal).stream().map(Token::value).collect(Collectors.joining());
      literalsByValue.computeIfAbsent(valueWithQuotes, key -> new ArrayList<>()).add(literal);
    }
  }

  @Override
  public void visitDecorator(Decorator decorator) {
    // Ignore literals in decorators
  }

  @Override
  public void visitTypeAnnotation(TypeAnnotation typeAnnotation) {
    // Ignore literals in type annotations
  }
}
