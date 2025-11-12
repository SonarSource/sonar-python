/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package com.sonar.python.it;

import org.assertj.core.api.AbstractAssert;
import org.sonarqube.ws.Issues;
import java.util.Comparator;
import java.util.List;

public class IssueListAssert extends AbstractAssert<IssueListAssert, List<Issues.Issue>> {
  public IssueListAssert(List<Issues.Issue> actual) {
    super(actual, IssueListAssert.class);
  }

  public IssueListAssert hasSize(int expectedSize) {
    isNotNull();
    
    int actualSize = actual.size();
    if (actualSize != expectedSize) {
      failWithMessage("Expected list to have size <%d> but was <%d>.%s", 
          expectedSize, actualSize, formatFoundIssues());
    }
    
    return this;
  }

  public IssueListAssert containsIssue(int line, String ruleKey) {
    isNotNull();
    
    boolean found = actual.stream()
        .anyMatch(issue -> issue.getLine() == line && ruleKey.equals(issue.getRule()));
    
    if (!found) {
      failWithMessage("Expected list to contain issue at line <%d> with rule <%s> but it was not found.%s", 
          line, ruleKey, formatFoundIssues());
    }
    
    return this;
  }

  public IssueListAssert doesNotContainIssue(int line, String ruleKey) {
    isNotNull();
    
    boolean found = actual.stream()
        .anyMatch(issue -> issue.getLine() == line && ruleKey.equals(issue.getRule()));
    
    if (found) {
      failWithMessage("Expected list to not contain issue at line <%d> with rule <%s> but it was found.%s", 
          line, ruleKey, formatFoundIssues());
    }
    
    return this;
  }

  private String formatFoundIssues() {
    if (actual.isEmpty()) {
      return " Found no issues.";
    }
    
    StringBuilder sb = new StringBuilder();
    sb.append(" Found issues:");
    actual.stream()
        .sorted(issueComparator())
        .forEach(issue -> sb.append(String.format("%n  - line %d: %s", issue.getLine(), issue.getRule())));
    
    return sb.toString();
  }

  private static Comparator<Issues.Issue> issueComparator() {
    return Comparator.comparing(Issues.Issue::getComponent)
        .thenComparing(Issues.Issue::getLine)
        .thenComparing(Issues.Issue::getRule);
  }

  public static IssueListAssert assertThat(List<Issues.Issue> actual) {
    return new IssueListAssert(actual);
  }
}
