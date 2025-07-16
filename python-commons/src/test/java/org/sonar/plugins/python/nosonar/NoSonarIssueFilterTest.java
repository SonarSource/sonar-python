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

import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.mockito.Mockito;
import org.sonar.api.rule.RuleKey;
import org.sonar.api.scan.issue.filter.FilterableIssue;
import org.sonar.api.scan.issue.filter.IssueFilterChain;
import org.sonar.plugins.python.api.nosonar.NoSonarLineInfo;

class NoSonarIssueFilterTest {
  
  @ParameterizedTest
  @MethodSource("provideFilterParameters")
  void test(Map<Integer, NoSonarLineInfo> noSonarInfos, @Nullable Integer line, boolean filterChainAcceptResult, boolean expectedResult) {
    var componentKey = "foo.py";
    var ruleRepo = "my_repo";
    var ruleKey = "my_rule";

    var collector = Mockito.mock(NoSonarLineInfoCollector.class);
    Mockito.when(collector.get(componentKey))
      .thenReturn(noSonarInfos);
    var filter = new NoSonarIssueFilter(collector);

    var issue = Mockito.mock(FilterableIssue.class);
    Mockito.when(issue.componentKey()).thenReturn(componentKey);
    Mockito.when(issue.line()).thenReturn(line);
    Mockito.when(issue.ruleKey()).thenReturn(RuleKey.of(ruleRepo, ruleKey));

    var filterChain = Mockito.mock(IssueFilterChain.class);
    Mockito.when(filterChain.accept(issue)).thenReturn(filterChainAcceptResult);

    var result = filter.accept(issue, filterChain);
    Assertions.assertThat(result).isEqualTo(expectedResult);
  }

  private static Stream<Arguments> provideFilterParameters() {
    return Stream.of(
      Arguments.of(Map.of(1, new NoSonarLineInfo(Set.of("my_rule"))), 1, true, false),
      Arguments.of(Map.of(1, new NoSonarLineInfo(Set.of("other_rule"))), 1, true, true),
      Arguments.of(Map.of(1, new NoSonarLineInfo(Set.of())), 1, true, true),
      Arguments.of(Map.of(2, new NoSonarLineInfo(Set.of("my_rule"))), 1, true, true),
      Arguments.of(Map.of(1, new NoSonarLineInfo(Set.of("my_rule"))), null, true, true),
      Arguments.of(Map.of(1, new NoSonarLineInfo(Set.of("other_rule"))), 1, false, false),
      Arguments.of(Map.of(), 1, false, false)
    );
  }
}
