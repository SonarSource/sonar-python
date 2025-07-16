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
package com.sonar.python.it;

import org.junit.jupiter.api.Test;
import org.sonarqube.ws.Issues;
import java.util.Collections;
import java.util.List;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class IssueListAssertTest {
  @Test
  void hasSize_should_pass_when_size_matches() {
    Issues.Issue issue1 = createIssue(10, "S1234", "file1");
    Issues.Issue issue2 = createIssue(20, "S5678", "file1");
    List<Issues.Issue> issues = List.of(issue1, issue2);
    
    IssueListAssert assertion = IssueListAssert.assertThat(issues);
    assertion.hasSize(2);
  }
  
  @Test
  void hasSize_should_fail_when_size_differs() {
    Issues.Issue issue = createIssue(10, "S1234", "file1");
    List<Issues.Issue> issues = List.of(issue);
    IssueListAssert assertion = IssueListAssert.assertThat(issues);
    
    assertThatThrownBy(() -> assertion.hasSize(2))
      .isInstanceOf(AssertionError.class)
      .hasMessageContaining("Expected list to have size <2> but was <1>")
      .hasMessageContaining("Found issues:")
      .hasMessageContaining("line 10: S1234");
  }
  
  @Test
  void containsIssue_should_pass_when_issue_matches() {
    Issues.Issue issue = createIssue(10, "S1234", "file1");
    List<Issues.Issue> issues = List.of(issue);
    
    IssueListAssert assertion = IssueListAssert.assertThat(issues);
    assertion.containsIssue(10, "S1234");
  }
  
  @Test
  void containsIssue_should_fail_when_line_differs() {
    Issues.Issue issue = createIssue(10, "S1234", "file1");
    List<Issues.Issue> issues = List.of(issue);
    IssueListAssert assertion = IssueListAssert.assertThat(issues);
    
    assertThatThrownBy(() -> assertion.containsIssue(20, "S1234"))
      .isInstanceOf(AssertionError.class)
      .hasMessageContaining("Expected list to contain issue at line <20> with rule <S1234> but it was not found")
      .hasMessageContaining("Found issues:")
      .hasMessageContaining("line 10: S1234");
  }
  
  @Test
  void containsIssue_should_fail_when_rule_differs() {
    Issues.Issue issue = createIssue(10, "S1234", "file1");
    List<Issues.Issue> issues = List.of(issue);
    IssueListAssert assertion = IssueListAssert.assertThat(issues);
    
    assertThatThrownBy(() -> assertion.containsIssue(10, "S5678"))
      .isInstanceOf(AssertionError.class)
      .hasMessageContaining("Expected list to contain issue at line <10> with rule <S5678> but it was not found")
      .hasMessageContaining("Found issues:")
      .hasMessageContaining("line 10: S1234");
  }
  
  @Test
  void containsIssue_should_fail_when_empty() {
    List<Issues.Issue> issues = Collections.emptyList();
    IssueListAssert assertion = IssueListAssert.assertThat(issues);
    
    assertThatThrownBy(() -> assertion.containsIssue(10, "S1234"))
      .isInstanceOf(AssertionError.class)
      .hasMessageContaining("Expected list to contain issue at line <10> with rule <S1234> but it was not found")
      .hasMessageContaining("Found no issues");
  }
  
  @Test
  void doesNotContainIssue_should_pass_when_issue_not_found() {
    Issues.Issue issue = createIssue(10, "S1234", "file1");
    List<Issues.Issue> issues = List.of(issue);
    
    IssueListAssert assertion = IssueListAssert.assertThat(issues);
    assertion.doesNotContainIssue(20, "S1234");
    assertion.doesNotContainIssue(10, "S5678");
  }
  
  @Test
  void doesNotContainIssue_should_fail_when_issue_found() {
    Issues.Issue issue = createIssue(10, "S1234", "file1");
    List<Issues.Issue> issues = List.of(issue);
    IssueListAssert assertion = IssueListAssert.assertThat(issues);
    
    assertThatThrownBy(() -> assertion.doesNotContainIssue(10, "S1234"))
      .isInstanceOf(AssertionError.class)
      .hasMessageContaining("Expected list to not contain issue at line <10> with rule <S1234> but it was found")
      .hasMessageContaining("Found issues:")
      .hasMessageContaining("line 10: S1234");
  }
  
  @Test
  void doesNotContainIssue_should_pass_when_empty() {
    List<Issues.Issue> issues = Collections.emptyList();
    IssueListAssert assertion = IssueListAssert.assertThat(issues);
    
    assertion.doesNotContainIssue(10, "S1234");
  }
  
  @Test
  void chained_assertions() {
    Issues.Issue issue1 = createIssue(10, "S1234", "file1");
    Issues.Issue issue2 = createIssue(20, "S5678", "file1");
    List<Issues.Issue> issues = List.of(issue1, issue2);
    
    IssueListAssert.assertThat(issues)
      .hasSize(2)
      .containsIssue(10, "S1234")
      .containsIssue(20, "S5678")
      .doesNotContainIssue(30, "S9999");
  }
  
  @Test
  void error_message_should_show_sorted_issues() {
    Issues.Issue issue1 = createIssue(20, "S5678", "file1");
    Issues.Issue issue2 = createIssue(10, "S1234", "file1");
    Issues.Issue issue3 = createIssue(10, "S9999", "file1");
    List<Issues.Issue> issues = List.of(issue1, issue2, issue3);
    IssueListAssert assertion = IssueListAssert.assertThat(issues);
    
    assertThatThrownBy(() -> assertion.containsIssue(5, "S0000"))
      .isInstanceOf(AssertionError.class)
      .hasMessageContaining("line 10: S1234")
      .hasMessageContaining("line 10: S9999")
      .hasMessageContaining("line 20: S5678");
  }
  
  private Issues.Issue createIssue(int line, String rule, String component) {
    Issues.Issue issue = mock(Issues.Issue.class);
    when(issue.getLine()).thenReturn(line);
    when(issue.getRule()).thenReturn(rule);
    when(issue.getComponent()).thenReturn(component);
    return issue;
  }
}
