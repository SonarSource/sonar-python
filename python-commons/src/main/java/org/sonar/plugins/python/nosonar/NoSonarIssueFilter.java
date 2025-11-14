/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import java.util.Set;
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

  private static final Set<String> WHITELISTED_RULES = Set.of("S1309", "NoSonar");

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
    String ruleId = issue.ruleKey().rule();

    if(WHITELISTED_RULES.contains(ruleId)) {
      LOG.debug("Rule {} cannot be filtered out as it is whitelisted (all whitelisted rules: {}) for component with key: {} on line: {}", 
        ruleId, 
        WHITELISTED_RULES,
        issueComponentKey, 
        issueLine);

      // while whitelisted rules should never be filtered out, returning true here screws up the LITs plugin
      return chain.accept(issue); 
    }

    var isNotFilteredOutByNoSonar = noSonarLineInfo == null || 
      (!noSonarLineInfo.isSuppressedRuleKeysEmpty() && !noSonarLineInfo.suppressedRuleKeys().contains(ruleId));
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
