/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.EnumMap;
import java.util.List;
import java.util.function.Consumer;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.FileInput;
import org.sonar.python.api.tree.Token;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.Tree.Kind;

public class SubscriptionVisitor {

  private final EnumMap<Kind, List<SubscriptionContextImpl>> consumers = new EnumMap<>(Kind.class);
  private final PythonVisitorContext pythonVisitorContext;
  private Tree currentElement;

  public static void analyze(Collection<PythonSubscriptionCheck> checks, PythonVisitorContext pythonVisitorContext) {
    SubscriptionVisitor subscriptionVisitor = new SubscriptionVisitor(checks, pythonVisitorContext);
    FileInput rootTree = pythonVisitorContext.rootTree();
    if (rootTree != null) {
      subscriptionVisitor.scan(rootTree);
      checks.forEach(PythonSubscriptionCheck::leaveFile);
    }
  }

  private SubscriptionVisitor(Collection<PythonSubscriptionCheck> checks, PythonVisitorContext pythonVisitorContext) {
    this.pythonVisitorContext = pythonVisitorContext;
    for (PythonSubscriptionCheck check : checks) {
      check.initialize((elementType, consumer) -> {
        List<SubscriptionContextImpl> elementConsumers = consumers.computeIfAbsent(elementType, c -> new ArrayList<>());
        elementConsumers.add(new SubscriptionContextImpl(check, consumer));
      });
    }
  }

  private void scan(Tree element) {
    Deque<Tree> stack = new ArrayDeque<>();
    stack.push(element);
    while (!stack.isEmpty()) {
      currentElement = stack.poll();
      consumers.getOrDefault(currentElement.getKind(), Collections.emptyList()).forEach(SubscriptionContextImpl::execute);
      for (int i = currentElement.children().size() - 1; i >= 0; i--) {
        if (currentElement.children().get(i) != null) {
          stack.push(currentElement.children().get(i));
        }
      }
    }
  }

  private class SubscriptionContextImpl implements SubscriptionContext {
    private final PythonCheck check;
    private final Consumer<SubscriptionContext> consumer;

    SubscriptionContextImpl(PythonCheck check, Consumer<SubscriptionContext> consumer) {
      this.check = check;
      this.consumer = consumer;
    }

    public void execute() {
      consumer.accept(this);
    }

    @Override
    public Tree syntaxNode() {
      return SubscriptionVisitor.this.currentElement;
    }

    @Override
    public PythonCheck.PreciseIssue addIssue(Tree element, @Nullable String message) {
      return addIssue(IssueLocation.preciseLocation(element, message));
    }

    @Override
    public PythonCheck.PreciseIssue addIssue(Token token, @Nullable String message) {
      return addIssue(IssueLocation.preciseLocation(token, message));
    }

    @Override
    public PythonCheck.PreciseIssue addIssue(Token from, Token to, @Nullable String message) {
      return addIssue(IssueLocation.preciseLocation(from, to, message));
    }

    @Override
    public PythonCheck.PreciseIssue addFileIssue(String message) {
      return addIssue(IssueLocation.atFileLevel(message));
    }

    @Override
    public PythonCheck.PreciseIssue addLineIssue(String message, int lineNumber) {
      return addIssue(IssueLocation.atLineLevel(message, lineNumber));
    }

    private PythonCheck.PreciseIssue addIssue(IssueLocation issueLocation) {
      PythonCheck.PreciseIssue newIssue = new PythonCheck.PreciseIssue(check, issueLocation);
      pythonVisitorContext.addIssue(newIssue);
      return newIssue;
    }

    @Override
    public PythonFile pythonFile() {
      return pythonVisitorContext.pythonFile();
    }
  }
}
