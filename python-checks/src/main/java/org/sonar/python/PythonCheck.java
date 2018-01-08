/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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
package org.sonar.python;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Token;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

public abstract class PythonCheck extends PythonVisitor {

  private List<PreciseIssue> issues = new ArrayList<>();

  public List<PreciseIssue> scanFileForIssues(PythonVisitorContext context) {
    issues.clear();
    scanFile(context);
    return Collections.unmodifiableList(new ArrayList<>(issues));
  }

  protected final PreciseIssue addIssue(AstNode node, String message) {
    PreciseIssue newIssue = new PreciseIssue(IssueLocation.preciseLocation(node, message));
    issues.add(newIssue);
    return newIssue;
  }

  protected final PreciseIssue addIssue(IssueLocation primaryLocation) {
    PreciseIssue newIssue = new PreciseIssue(primaryLocation);
    issues.add(newIssue);
    return newIssue;
  }

  protected final PreciseIssue addLineIssue(String message, int lineNumber) {
    PreciseIssue newIssue = new PreciseIssue(IssueLocation.atLineLevel(message, lineNumber));
    issues.add(newIssue);
    return newIssue;
  }

  protected final PreciseIssue addFileIssue(String message) {
    PreciseIssue newIssue = new PreciseIssue(IssueLocation.atFileLevel(message));
    issues.add(newIssue);
    return newIssue;
  }

  protected final PreciseIssue addIssue(Token token, String message) {
    return addIssue(new AstNode(token), message);
  }

  public static class PreciseIssue {

    private final IssueLocation primaryLocation;
    private Integer cost;
    private final List<IssueLocation> secondaryLocations;

    private PreciseIssue(IssueLocation primaryLocation) {
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

    public PreciseIssue secondary(AstNode node, @Nullable String message) {
      secondaryLocations.add(IssueLocation.preciseLocation(node, message));
      return this;
    }

    public PreciseIssue secondary(IssueLocation issueLocation) {
      secondaryLocations.add(issueLocation);
      return this;
    }

    public List<IssueLocation> secondaryLocations() {
      return secondaryLocations;
    }
  }

  public static <T> Set<T> immutableSet(T... el) {
    return Collections.unmodifiableSet(new HashSet<>(Arrays.asList(el)));
  }
}
