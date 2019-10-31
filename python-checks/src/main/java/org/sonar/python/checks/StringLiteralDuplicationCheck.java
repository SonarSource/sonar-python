/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonVisitorCheck;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.StringElement;

@Rule(key = "S1192")
public class StringLiteralDuplicationCheck extends PythonVisitorCheck {

  private static final Integer MINIMUM_LITERAL_LENGTH = 5;
  private static final int DEFAULT_THRESHOLD = 3;
  private static final Pattern EXCLUSION_PATTERN = Pattern.compile("[_a-zA-Z0-9]+");

  @RuleProperty(
    key = "threshold",
    description = "Number of times a literal must be duplicated to trigger an issue",
    defaultValue = "" + DEFAULT_THRESHOLD)
  public int threshold = DEFAULT_THRESHOLD;

  private Map<String, List<StringElement>> literalsByValue = new HashMap<>();

  @Override
  public void visitFileInput(FileInput fileInput) {
    literalsByValue.clear();

    super.visitFileInput(fileInput);

    for (Map.Entry<String, List<StringElement>> entry : literalsByValue.entrySet()) {
      List<StringElement> occurrences = entry.getValue();
      int nbOfOccurrences = occurrences.size();
      if (nbOfOccurrences >= threshold) {
        StringElement first = occurrences.get(0);
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
  public void visitStringElement(StringElement literal) {
    String value = literal.trimmedQuotesValue();
    if (value.length() >= MINIMUM_LITERAL_LENGTH && !literal.isInterpolated() && !EXCLUSION_PATTERN.matcher(value).matches()) {
      literalsByValue.computeIfAbsent(literal.firstToken().value(), key -> new ArrayList<>()).add(literal);
    }
  }

  @Override
  public void visitDecorator(Decorator decorator) {
    // Ignore literals in decorators
  }
}
