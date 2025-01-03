/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.regex;

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonarsource.analyzer.commons.regex.RegexIssueLocation;
import org.sonarsource.analyzer.commons.regex.ast.IndexRange;
import org.sonarsource.analyzer.commons.regex.ast.RegexSyntaxElement;

public class PythonRegexIssueLocation {

  private PythonRegexIssueLocation() {

  }

  public static IssueLocation preciseLocation(RegexIssueLocation regexIssueLocation) {
    return preciseLocation(regexIssueLocation.syntaxElements(), regexIssueLocation.message());
  }

  public static IssueLocation preciseLocation(RegexSyntaxElement syntaxElement, String message) {
    return preciseLocation(Collections.singletonList(syntaxElement), message);
  }

  public static IssueLocation preciseLocation(List<RegexSyntaxElement> syntaxElements, String message) {
    RegexSyntaxElement firstElement = syntaxElements.get(0);
    PythonAnalyzerRegexSource source = (PythonAnalyzerRegexSource) firstElement.getSource();
    IndexRange current = firstElement.getRange();

    for (RegexSyntaxElement syntaxElement : syntaxElements.subList(1, syntaxElements.size())) {
      if (syntaxElement.getRange().getBeginningOffset() == current.getEndingOffset()) {
        current = new IndexRange(current.getBeginningOffset(), syntaxElement.getRange().getEndingOffset());
      }
      // We do not combine RegexSyntaxElement which are not located side by side
    }
    return IssueLocation.preciseLocation(source.locationInFileFor(current), message);
  }
}
