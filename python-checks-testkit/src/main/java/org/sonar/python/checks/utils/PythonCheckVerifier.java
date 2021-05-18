/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
import java.io.File;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonCheck.PreciseIssue;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Trivia;
import org.sonar.python.SubscriptionVisitor;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.tree.TreeUtils;
import org.sonarsource.analyzer.commons.checks.verifier.MultiFileVerifier;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.sonar.python.semantic.SymbolUtils.pythonPackageName;

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
    verify(Collections.singletonList(path), check);
  }

  public static void verifyNoIssue(String path, PythonCheck check) {
    File file = new File(path);
    createVerifier(Collections.singletonList(file), check, ProjectLevelSymbolTable.empty(), null).assertNoIssues();
  }

  public static void verify(List<String> paths, PythonCheck check) {
    List<File> files = paths.stream().map(File::new).collect(Collectors.toList());
    File baseDirFile = new File(files.get(0).getParent());
    ProjectLevelSymbolTable projectLevelSymbolTable = TestPythonVisitorRunner.globalSymbols(files, baseDirFile);
    createVerifier(files, check, projectLevelSymbolTable, baseDirFile).assertOneOrMoreIssues();
  }

  public static void verifyNoIssue(List<String> paths, PythonCheck check) {
    List<File> files = paths.stream().map(File::new).collect(Collectors.toList());
    File baseDirFile = new File(files.get(0).getParent());
    ProjectLevelSymbolTable projectLevelSymbolTable = TestPythonVisitorRunner.globalSymbols(files, baseDirFile);
    createVerifier(files, check, projectLevelSymbolTable, baseDirFile).assertNoIssues();
  }

  private static MultiFileVerifier createVerifier(List<File> files, PythonCheck check, ProjectLevelSymbolTable projectLevelSymbolTable, @Nullable File baseDir) {
    MultiFileVerifier multiFileVerifier = MultiFileVerifier.create(files.get(0).toPath(), UTF_8);
    for (File file : files) {
      PythonVisitorContext context = baseDir != null
        ? TestPythonVisitorRunner.createContext(file, null, pythonPackageName(file, baseDir.getAbsolutePath()), projectLevelSymbolTable)
        : TestPythonVisitorRunner.createContext(file);
      addFileIssues(check, multiFileVerifier, file, context);
    }
    return multiFileVerifier;
  }

  private static void addFileIssues(PythonCheck check, MultiFileVerifier multiFileVerifier, File file, PythonVisitorContext context) {
    for (PreciseIssue issue : scanFileForIssues(check, context)) {
      if (!issue.check().equals(check)) {
        throw new IllegalStateException("Verifier support only one kind of issue " + issue.check() + " != " + check);
      }
      Integer cost = issue.cost();
      addPreciseIssue(file.toPath(), multiFileVerifier, issue).withGap(cost == null ? null : (double) cost);
    }

    for (Token token : TreeUtils.tokens(context.rootTree())) {
      for (Trivia trivia : token.trivia()) {
        multiFileVerifier.addComment(file.toPath(), trivia.token().line(), trivia.token().column() + 1, trivia.value(), 1, 0);
      }
    }
  }

  private static MultiFileVerifier.Issue addPreciseIssue(Path path, MultiFileVerifier verifier, PreciseIssue preciseIssue) {
    IssueLocation location = preciseIssue.primaryLocation();
    String message = location.message();
    Preconditions.checkNotNull(message, "Primary location message should never be null.");

    if (location.startLine() == IssueLocation.UNDEFINED_LINE) {
      return verifier.reportIssue(path, message).onFile();
    }

    if (location.startLineOffset() == IssueLocation.UNDEFINED_OFFSET) {
      return verifier.reportIssue(path, message).onLine(location.startLine());
    }


    MultiFileVerifier.Issue issueBuilder = verifier.reportIssue(path, message)
      .onRange(location.startLine(), location.startLineOffset() + 1, location.endLine(), location.endLineOffset());
    for (IssueLocation secondary : preciseIssue.secondaryLocations()) {
      issueBuilder.addSecondary(path, secondary.startLine(), secondary.startLineOffset() + 1, secondary.endLine(), secondary.endLineOffset(), secondary.message());
    }
    return issueBuilder;
  }

}
