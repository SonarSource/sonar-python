/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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

import com.google.common.collect.ImmutableList;
import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Grammar;
import com.sonar.sslr.api.Token;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;
import org.sonar.squidbridge.checks.SquidCheck;

public abstract class PythonCheck extends SquidCheck<Grammar> {

  private List<PreciseIssue> issues = new ArrayList<>();

  public List<PreciseIssue> getIssues() {
    return ImmutableList.copyOf(issues);
  }

  protected final PreciseIssue addIssue(AstNode node, String message) {
    PreciseIssue newIssue = new PreciseIssue(new IssueLocation(node, message), getContext().getFile());
    issues.add(newIssue);
    return newIssue;
  }

  protected final PreciseIssue addIssue(IssueLocation primaryLocation) {
    PreciseIssue newIssue = new PreciseIssue(primaryLocation, getContext().getFile());
    issues.add(newIssue);
    return newIssue;
  }

  protected final PreciseIssue addIssue(Token token, String message) {
    return addIssue(new AstNode(token), message);
  }

  public static class PreciseIssue {

    private File file;
    private IssueLocation primaryLocation;
    private Double cost;
    private List<IssueLocation> secondaryLocations;

    private PreciseIssue(IssueLocation primaryLocation, File file) {
      this.primaryLocation = primaryLocation;
      this.secondaryLocations = new ArrayList<>();
      this.file = file;
    }

    @Nullable
    public Double cost() {
      return cost;
    }

    public PreciseIssue cost(double cost) {
      this.cost = cost;
      return this;
    }

    public File file() {
      return file;
    }

    public IssueLocation primaryLocation() {
      return primaryLocation;
    }

    public PreciseIssue secondary(AstNode node, @Nullable String message) {
      secondaryLocations.add(new IssueLocation(node, message));
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

  public static class IssueLocation {
    private Token firstToken;
    private TokenLocation lastTokenLocation;
    private String message;

    public IssueLocation(AstNode node, String message) {
      this.message = message;
      this.firstToken = node.getToken();
      this.lastTokenLocation = new TokenLocation(node.getLastToken());
    }

    public IssueLocation(AstNode startNode, AstNode endNode, String message) {
      this.message = message;
      this.firstToken = startNode.getToken();
      this.lastTokenLocation = new TokenLocation(endNode.getLastToken());
    }

    public String message() {
      return message;
    }

    public int startLine() {
      return firstToken.getLine();
    }

    public int startLineOffset() {
      return firstToken.getColumn();
    }

    public int endLine() {
      return lastTokenLocation.endLine();
    }

    public int endLineOffset() {
      return lastTokenLocation.endLineOffset();
    }
  }
}
