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
package org.sonar.plugins.python;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;
import javax.annotation.Nullable;
import org.sonar.api.batch.rule.Severity;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.batch.sensor.issue.Issue;
import org.sonar.api.batch.sensor.issue.IssueLocation;
import org.sonar.api.batch.sensor.issue.NewIssue;
import org.sonar.api.batch.sensor.issue.NewIssueLocation;
import org.sonar.api.batch.sensor.issue.fix.NewQuickFix;
import org.sonar.api.batch.sensor.issue.fix.QuickFix;
import org.sonar.api.issue.impact.SoftwareQuality;
import org.sonar.api.rule.RuleKey;
import org.sonarsource.sonarlint.core.analysis.sonarapi.DefaultSonarLintIssue;

// This class was copied from sonar-java
class MockSonarLintIssue implements NewIssue, Issue {
  private final DefaultSonarLintIssue parent = new DefaultSonarLintIssue(null, null, null);
  private final SensorContextTester context;
  private boolean isQuickFixAvailable = false;
  @Nullable
  private List<String> codeVariants = null;
  private boolean saved;

  MockSonarLintIssue(SensorContextTester context) {
    this.context = context;
  }

  @Override
  public NewIssue addQuickFix(NewQuickFix newQuickFix) {
    return parent.addQuickFix(newQuickFix);
  }

  @Override
  public NewQuickFix newQuickFix() {
    return parent.newQuickFix();
  }

  @Override
  public NewIssue forRule(RuleKey ruleKey) {
    parent.forRule(ruleKey);
    return this;
  }

  @Override
  public NewIssue gap(Double gap) {
    parent.gap(gap);
    return this;
  }

  @Override
  public NewIssue overrideSeverity(Severity severity) {
    throw new IllegalStateException("Not supposed to be tested");
  }

  @Override
  public NewIssue overrideImpact(SoftwareQuality softwareQuality, org.sonar.api.issue.impact.Severity severity) {
    throw new IllegalStateException("Not supposed to be tested");
  }

  @Override
  public NewIssue at(NewIssueLocation primaryLocation) {
    parent.at(primaryLocation);
    return this;
  }

  @Override
  public NewIssue addLocation(NewIssueLocation secondaryLocation) {
    throw new IllegalStateException("Not supposed to be tested");
  }

  @Override
  public NewIssue setQuickFixAvailable(boolean b) {
    isQuickFixAvailable = b;
    return this;
  }

  @Override
  public NewIssue addFlow(Iterable<NewIssueLocation> flowLocations) {
    throw new IllegalStateException("Not supposed to be tested");
  }

  @Override
  public NewIssue addFlow(Iterable<NewIssueLocation> iterable, FlowType flowType, @Nullable String s) {
    return null;
  }

  @Override
  public NewIssueLocation newLocation() {
    return parent.newLocation();
  }

  @Override
  public void save() {
    // save() is final in DefaultSonarLintIssue
    this.saved = true;
    context.allIssues().add(this);
  }

  @Override
  public RuleKey ruleKey() {
    throw new IllegalStateException("Not supposed to be tested");
  }

  @Override
  public Double gap() {
    throw new IllegalStateException("Not supposed to be tested");
  }

  @Override
  public Severity overriddenSeverity() {
    throw new IllegalStateException("Not supposed to be tested");
  }

  @Override
  public Map<SoftwareQuality, org.sonar.api.issue.impact.Severity> overridenImpacts() {
    throw new IllegalStateException("Not supposed to be tested");
  }

  @Override
  public IssueLocation primaryLocation() {
    throw new IllegalStateException("Not supposed to be tested");
  }

  @Override
  public List<Flow> flows() {
    throw new IllegalStateException("Not supposed to be tested");
  }

  @Override
  public boolean isQuickFixAvailable() {
    return isQuickFixAvailable;
  }

  @Override
  public Optional<String> ruleDescriptionContextKey() {
    return Optional.empty();
  }

  @Override
  public List<QuickFix> quickFixes() {
    return parent.quickFixes();
  }

  @Nullable
  @Override
  public List<String> codeVariants() {
    return codeVariants;
  }

  @Override
  public NewIssue setRuleDescriptionContextKey(String ruleDescriptionContextKey) {
    return this;
  }

  @Override
  public NewIssue setCodeVariants(@Nullable Iterable<String> iterable) {
    codeVariants = iterable == null ? null : StreamSupport.stream(iterable.spliterator(), false)
      .toList();
    return this;
  }

  public boolean getSaved() {
    return saved;
  }
}
