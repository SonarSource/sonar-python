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
package org.sonar.plugins.python.nosonar;

import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.api.rule.RuleKey;
import org.sonar.api.scan.issue.filter.FilterableIssue;
import org.sonar.api.scan.issue.filter.IssueFilterChain;
import org.sonar.plugins.python.api.nosonar.NoSonarLineInfo;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class NoSonarIssueFilterTest {

  private static final Set<String> SECURITY_RULES = Set.of("S4423");

  @ParameterizedTest
  @MethodSource("provideFilterParameters")
  void test(Map<Integer, NoSonarLineInfo> noSonarInfos, String ruleKey, @Nullable Integer line, boolean filterChainAcceptResult, boolean expectedResult) {
    var componentKey = "foo.py";
    var ruleRepo = "my_repo";

    var collector = mock(NoSonarLineInfoCollector.class);
    when(collector.get(componentKey))
      .thenReturn(noSonarInfos);

    var filter = new NoSonarIssueFilter(collector, new SecurityRuleKeyProvider(SECURITY_RULES));

    var issue = mock(FilterableIssue.class);
    when(issue.componentKey()).thenReturn(componentKey);
    when(issue.line()).thenReturn(line);
    when(issue.ruleKey()).thenReturn(RuleKey.of(ruleRepo, ruleKey));

    var filterChain = mock(IssueFilterChain.class);
    when(filterChain.accept(issue)).thenReturn(filterChainAcceptResult);

    var result = filter.accept(issue, filterChain);
    Assertions.assertThat(result)
      .as("NoSonarIssueFilter filter result")
      .isEqualTo(expectedResult);
  }

  private static Stream<Arguments> provideFilterParameters() {
    return Stream.concat(
      provideBaseFilterParameters(),
      provideSecurityOnlySuppressionParameters()
    );
  }

  private static Stream<Arguments> provideBaseFilterParameters() {
    return Stream.of(
      // explicit key suppression: suppress matching rule
      Arguments.of(Map.of(1, new NoSonarLineInfo(Set.of("my_rule"))), "my_rule", 1, true, false),
      // bare NOSONAR/noqa (securityOnlySuppression=false): suppress all
      Arguments.of(Map.of(1, new NoSonarLineInfo(Set.of())), "my_rule", 1, true, false),
      Arguments.of(Map.of(1, new NoSonarLineInfo(Set.of())), "my_rule", 1, false, false),
      // explicit key suppression: different rule, not suppressed
      Arguments.of(Map.of(1, new NoSonarLineInfo(Set.of("other_rule"))), "my_rule", 1, true, true),
      // suppression on different line: not suppressed
      Arguments.of(Map.of(2, new NoSonarLineInfo(Set.of("my_rule"))), "my_rule", 1, true, true),
      // issue has no line: never suppressed
      Arguments.of(Map.of(1, new NoSonarLineInfo(Set.of("my_rule"))), "my_rule", null, true, true),
      // different rule on same line with chain rejecting: chain result propagates
      Arguments.of(Map.of(1, new NoSonarLineInfo(Set.of("other_rule"))), "my_rule", 1, false, false),
      // S1309 is whitelisted — never suppressed by explicit key
      Arguments.of(Map.of(1, new NoSonarLineInfo(Set.of("S1309"))), "S1309", 1, true, true),
      Arguments.of(Map.of(1, new NoSonarLineInfo(Set.of("S1309"))), "S1309", 1, false, false),
      // no suppression comment: chain decides
      Arguments.of(Map.of(), "my_rule", 1, false, false)
    );
  }

  private static Stream<Arguments> provideSecurityOnlySuppressionParameters() {
    return Stream.of(
      // bare nosec: suppresses security rule S4423
      Arguments.of(Map.of(1, NoSonarLineInfo.securityOnly()), "S4423", 1, true, false),
      // bare nosec: does NOT suppress non-security rule
      Arguments.of(Map.of(1, NoSonarLineInfo.securityOnly()), "my_rule", 1, true, true),
      // bare nosec: S1309 is whitelisted, chain decides regardless
      Arguments.of(Map.of(1, NoSonarLineInfo.securityOnly()), "S1309", 1, true, true),
      Arguments.of(Map.of(1, NoSonarLineInfo.securityOnly()), "S1309", 1, false, false),
      // mixed: explicit key + nosec — explicit key is suppressed
      Arguments.of(Map.of(1, new NoSonarLineInfo(Set.of("E501"), "", true)), "E501", 1, true, false),
      // mixed: explicit key + nosec — security rule is also suppressed
      Arguments.of(Map.of(1, new NoSonarLineInfo(Set.of("E501"), "", true)), "S4423", 1, true, false),
      // mixed: explicit key + nosec — non-security rule not in keys is NOT suppressed
      Arguments.of(Map.of(1, new NoSonarLineInfo(Set.of("E501"), "", true)), "my_rule", 1, true, true)
    );
  }
}
