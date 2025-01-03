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
package org.sonar.plugins.python.api;

import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;
import org.sonar.api.Beta;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;

public interface PythonCheck {


  void scanFile(PythonVisitorContext visitorContext);

  @Beta
  default boolean scanWithoutParsing(PythonInputFileContext inputFileContext) {
    return true;
  }

  class PreciseIssue {

    private final PythonCheck check;
    private final IssueLocation primaryLocation;
    private Integer cost;
    private final List<IssueLocation> secondaryLocations;
    private final List<PythonQuickFix> quickFixes = new ArrayList<>();

    public PreciseIssue(PythonCheck check, IssueLocation primaryLocation) {
      this.check = check;
      this.primaryLocation = primaryLocation;
      this.secondaryLocations = new ArrayList<>();
    }

    @Nullable
    public Integer cost() {
      return cost;
    }

    public PreciseIssue withCost(int cost) {
      this.cost = cost;
      return this;
    }

    public IssueLocation primaryLocation() {
      return primaryLocation;
    }

    public PreciseIssue secondary(Tree tree, @Nullable String message) {
      secondaryLocations.add(IssueLocation.preciseLocation(tree, message));
      return this;
    }

    public PreciseIssue secondary(Token token, @Nullable String message) {
      secondaryLocations.add(IssueLocation.preciseLocation(token, message));
      return this;
    }

    public PreciseIssue secondary(IssueLocation issueLocation) {
      secondaryLocations.add(issueLocation);
      return this;
    }

    public PreciseIssue secondary(LocationInFile locationInFile, @Nullable String message) {
      secondaryLocations.add(IssueLocation.preciseLocation(locationInFile, message));
      return this;
    }

    public List<IssueLocation> secondaryLocations() {
      return secondaryLocations;
    }

    /**
     * This only makes sense in SonarLint context. Should not be used in custom rules.
     */
    @Beta
    public void addQuickFix(PythonQuickFix quickFix){
      this.quickFixes.add(quickFix);
    }

    public List<PythonQuickFix> quickFixes() {
      return quickFixes;
    }

    public PythonCheck check() {
      return check;
    }
  }

  enum CheckScope {
    MAIN,
    ALL
  }

  default CheckScope scope() {
    return CheckScope.MAIN;
  }
}
