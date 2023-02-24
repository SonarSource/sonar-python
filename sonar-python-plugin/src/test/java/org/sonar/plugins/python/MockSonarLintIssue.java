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
package org.sonar.plugins.python;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.api.batch.rule.Severity;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.batch.sensor.issue.Issue;
import org.sonar.api.batch.sensor.issue.IssueLocation;
import org.sonar.api.batch.sensor.issue.NewIssue;
import org.sonar.api.batch.sensor.issue.NewIssueLocation;
import org.sonar.api.rule.RuleKey;
import org.sonarsource.sonarlint.core.analysis.container.analysis.issue.SensorQuickFix;
import org.sonarsource.sonarlint.core.analysis.sonarapi.DefaultSonarLintIssue;
import org.sonarsource.sonarlint.plugin.api.issue.NewQuickFix;
import org.sonarsource.sonarlint.plugin.api.issue.NewSonarLintIssue;

// This class was copied from sonar-java
public class MockSonarLintIssue implements NewIssue, NewSonarLintIssue, Issue {
  private final DefaultSonarLintIssue parent = new DefaultSonarLintIssue(null, null, null);
  private final SensorContextTester context;
  public final List<SensorQuickFix> quickFixes = new ArrayList<>();
  private boolean isQuickFixAvailable = false;
  private boolean saved;

  MockSonarLintIssue(SensorContextTester context) {
    this.context = context;
  }

  @Override
  public NewQuickFix newQuickFix() {
    return parent.newQuickFix();
  }

  @Override
  public NewSonarLintIssue addQuickFix(NewQuickFix newQuickFix) {
    quickFixes.add((SensorQuickFix) newQuickFix);
    return parent.addQuickFix(newQuickFix);
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
  public NewIssue at(NewIssueLocation primaryLocation) {
    parent.at(primaryLocation);
    return this;
  }

  @Override
  public NewIssue addLocation(NewIssueLocation secondaryLocation) {
    throw new IllegalStateException("Not supposed to be tested");
  }

  // @Override in SonarQube 9.2
  public NewIssue setQuickFixAvailable(boolean b) {
    isQuickFixAvailable = b;
    return this;
  }

  @Override
  public NewIssue addFlow(Iterable<NewIssueLocation> flowLocations) {
    throw new IllegalStateException("Not supposed to be tested");
  }

  @Override
  public NewIssue addFlow(Iterable<NewIssueLocation> flowLocations, FlowType flowType, @Nullable String flowDescription) {
    throw new IllegalStateException("Not supposed to be tested");
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
  public NewIssue setRuleDescriptionContextKey(@Nullable String ruleDescriptionContextKey) {
    throw new IllegalStateException("Not supposed to be tested");
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
    throw new IllegalStateException("Not supposed to be tested");
  }

  protected boolean getSaved(){
    return saved;
  }

}
