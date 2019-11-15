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
package org.sonar.python.checks.utils;

import com.google.common.base.Preconditions;
import com.sonarsource.checks.verifier.SingleFileVerifier;
import java.io.File;
import java.util.Collections;
import java.util.List;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonCheck.PreciseIssue;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Trivia;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.python.SubscriptionVisitor;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.tree.TreeUtils;

import static java.nio.charset.StandardCharsets.UTF_8;

public class PythonCheckVerifier {

  private PythonCheckVerifier() {
  }

  private static List<PreciseIssue> scanFileForIssues(PythonCheck check, PythonVisitorContext context) {
    check.scanFile(context);
    if (check instanceof PythonSubscriptionCheck) {
      SubscriptionVisitor.analyze(Collections.singletonList((PythonSubscriptionCheck) check), context);
    }
    return context.getIssues();
  }


  public static void verify(String path, PythonCheck check) {
    File file = new File(path);
    createVerifier(file, check).assertOneOrMoreIssues();
  }

  public static void verifyNoIssue(String path, PythonCheck check) {
    File file = new File(path);
    createVerifier(file, check).assertNoIssues();
  }

  private static SingleFileVerifier createVerifier(File file, PythonCheck check) {
    SingleFileVerifier verifier = SingleFileVerifier.create(file.toPath(), UTF_8);
    return getSingleFileVerifier(check, verifier, file);
  }

  private static SingleFileVerifier getSingleFileVerifier(PythonCheck check, SingleFileVerifier verifier, File file) {
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(file);
    for (PreciseIssue issue : scanFileForIssues(check, context)) {
      if (!issue.check().equals(check)) {
        throw new IllegalStateException("Verifier support only one kind of issue " + issue.check() + " != " + check);
      }
      Integer cost = issue.cost();
      addPreciseIssue(verifier, issue).withGap(cost == null ? null : (double) cost);
    }

    for (Token token : TreeUtils.tokens(context.rootTree())) {
      for (Trivia trivia : token.trivia()) {
        verifier.addComment(trivia.token().line(), trivia.token().column() + 1, trivia.value(), 1, 0);
      }
    }
    return verifier;
  }

  private static SingleFileVerifier.Issue addPreciseIssue(SingleFileVerifier verifier, PreciseIssue preciseIssue) {
    IssueLocation location = preciseIssue.primaryLocation();
    String message = location.message();
    Preconditions.checkNotNull(message, "Primary location message should never be null.");

    if (location.startLine() == IssueLocation.UNDEFINED_LINE) {
      return verifier.reportIssue(message).onFile();
    }

    if (location.startLineOffset() == IssueLocation.UNDEFINED_OFFSET) {
      return verifier.reportIssue(message).onLine(location.startLine());
    }


    SingleFileVerifier.Issue issueBuilder = verifier.reportIssue(message)
      .onRange(location.startLine(), location.startLineOffset() + 1, location.endLine(), location.endLineOffset());
    for (IssueLocation secondary : preciseIssue.secondaryLocations()) {
      issueBuilder.addSecondary(secondary.startLine(), secondary.startLineOffset() + 1, secondary.endLine(), secondary.endLineOffset(), secondary.message());
    }
    return issueBuilder;
  }

}
