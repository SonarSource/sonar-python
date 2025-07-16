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
package org.sonar.plugins.python.nosonar;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.scan.issue.filter.FilterableIssue;
import org.sonar.api.scan.issue.filter.IssueFilter;
import org.sonar.api.scan.issue.filter.IssueFilterChain;
import org.sonar.api.scanner.ScannerSide;
import org.sonarsource.api.sonarlint.SonarLintSide;

@SonarLintSide
@ScannerSide
public class NoSonarIssueFilter implements IssueFilter {
  private static final Logger LOG = LoggerFactory.getLogger(NoSonarIssueFilter.class);

  private final NoSonarLineInfoCollector noSonarLineInfoCollector;

  public NoSonarIssueFilter(NoSonarLineInfoCollector noSonarLineInfoCollector) {
    this.noSonarLineInfoCollector = noSonarLineInfoCollector;
  }

  @Override
  public boolean accept(FilterableIssue issue, IssueFilterChain chain) {
    var issueLine = issue.line();
    if (issueLine == null) {
      return chain.accept(issue);
    }
    var issueComponentKey = issue.componentKey();

    var noSonarLineInfos = noSonarLineInfoCollector.get(issueComponentKey);
    var noSonarLineInfo = noSonarLineInfos.get(issueLine);
    var isNotFilteredOutByNoSonar = noSonarLineInfo == null || !noSonarLineInfo.suppressedRuleKeys().contains(issue.ruleKey().rule());
    if (!isNotFilteredOutByNoSonar) {
      LOG.debug("Filtering out issue in the component with key: {} for rule: {} on line: {} based on the file NoSonar infos {}",
        issueComponentKey,
        issue.ruleKey().rule(),
        issueLine,
        noSonarLineInfos.values());
    }
    return isNotFilteredOutByNoSonar && chain.accept(issue);
  }
}
