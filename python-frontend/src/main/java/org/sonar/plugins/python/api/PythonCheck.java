/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
package org.sonar.plugins.python.api;

import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.reporting.PythonQuickFix;

public interface PythonCheck {


  void scanFile(PythonVisitorContext visitorContext);

  class PreciseIssue {

    private final PythonCheck check;
    private final IssueLocation primaryLocation;
    private Integer cost;
    private final List<IssueLocation> secondaryLocations;
    // TODO NEED TO HIDE, should not be in the public api for now
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

    public void addQuickFix(PythonQuickFix quickFix){
      this.quickFixes.add(quickFix);
    }

    public List<PythonQuickFix> getQuickFixes() {
      return quickFixes;
    }

    public PythonCheck check() {
      return check;
    }
  }
}
