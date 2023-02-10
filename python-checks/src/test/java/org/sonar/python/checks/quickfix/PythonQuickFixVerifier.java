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
package org.sonar.python.checks.quickfix;

import com.sonar.sslr.api.AstNode;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonCheck.PreciseIssue;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.SubscriptionVisitor;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.parser.PythonParser;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.tree.PythonTreeMaker;

import static org.assertj.core.api.Assertions.assertThat;

public class PythonQuickFixVerifier {
  private PythonQuickFixVerifier() {
  }

  public static void verify(PythonCheck check, String codeWithIssue, String... codesFixed) {
    List<PythonCheck.PreciseIssue> issues = PythonQuickFixVerifier
      .getIssuesWithQuickFix(check, codeWithIssue);

    assertThat(issues)
      .as("Number of issues")
      .overridingErrorMessage("Expected 1 issue but found %d", issues.size())
      .hasSize(1);
    PreciseIssue issue = issues.get(0);

    assertThat(issue.quickFixes())
      .as("Number of quickfixes")
      .overridingErrorMessage("Expected %d quickfix but found %d", codesFixed.length, issue.quickFixes().size())
      .hasSize(codesFixed.length);

    List<String> appliedQuickFix = issue.quickFixes().stream()
      .map(quickFix -> applyQuickFix(codeWithIssue, quickFix))
      .collect(Collectors.toList());

    assertThat(appliedQuickFix)
      .as("The code with the quickfix applied is not the expected result.\n" +
        "\"Applied QuickFixes are:\n%s\nExpected result:\n%s", appliedQuickFix, Arrays.asList(codesFixed))
      .isEqualTo(Arrays.asList(codesFixed));
  }

  public static void verifyNoQuickFixes(PythonCheck check, String codeWithIssue) {
    List<PythonCheck.PreciseIssue> issues = PythonQuickFixVerifier
      .getIssuesWithQuickFix(check, codeWithIssue);

    assertThat(issues)
      .as("Number of issues")
      .overridingErrorMessage("Expected 1 issue but found %d", issues.size())
      .hasSize(1);
    PreciseIssue issue = issues.get(0);

    assertThat(issue.quickFixes())
      .as("Number of quick fixes")
      .overridingErrorMessage("Expected no quick fixes for the issue but found %d", issue.quickFixes().size())
      .isEmpty();
  }

  public static void verifyQuickFixMessages(PythonCheck check, String codeWithIssue, String... expectedMessages) {
    Stream<String> descriptions = PythonQuickFixVerifier
      .getIssuesWithQuickFix(check, codeWithIssue)
      .stream()
      .flatMap(issue -> issue.quickFixes().stream())
      .map(PythonQuickFix::getDescription);

    assertThat(descriptions).containsExactly(expectedMessages);
  }

  private static List<PreciseIssue> scanFileForIssues(PythonCheck check, PythonVisitorContext context) {
    check.scanFile(context);
    if (check instanceof PythonSubscriptionCheck) {
      SubscriptionVisitor.analyze(Collections.singletonList((PythonSubscriptionCheck) check), context);
    }
    return context.getIssues();
  }

  private static List<PreciseIssue> getIssuesWithQuickFix(PythonCheck check, String codeWithIssue) {
    PythonParser parser = PythonParser.create();
    PythonQuickFixFile pythonFile = new PythonQuickFixFile(codeWithIssue);
    AstNode astNode = parser.parse(pythonFile.content());
    FileInput parse = new PythonTreeMaker().fileInput(astNode);

    PythonVisitorContext visitorContext = new PythonVisitorContext(parse,
      pythonFile, null, "",
      ProjectLevelSymbolTable.empty(), CacheContextImpl.dummyCache());

    return scanFileForIssues(check, visitorContext);
  }

  private static String applyQuickFix(String codeWithIssue, PythonQuickFix quickFix) {
    List<PythonTextEdit> sortedEdits = sortTextEdits(quickFix.getTextEdits());
    String codeBeingFixed = codeWithIssue;
    for (PythonTextEdit edit : sortedEdits) {
      codeBeingFixed = applyTextEdit(codeBeingFixed, edit);
    }
    return codeBeingFixed;
  }

  private static String applyTextEdit(String codeWithIssue, PythonTextEdit textEdit) {
    String replacement = textEdit.replacementText();
    int start = convertPositionToIndex(codeWithIssue, textEdit.startLine(), textEdit.startLineOffset());
    int end = convertPositionToIndex(codeWithIssue, textEdit.endLine(), textEdit.endLineOffset());
    return codeWithIssue.substring(0, start) + replacement + codeWithIssue.substring(end);
  }

  private static List<PythonTextEdit> sortTextEdits(List<PythonTextEdit> pythonTextEdits) {
    checkNoCollision(pythonTextEdits);
    ArrayList<PythonTextEdit> list = new ArrayList<>(pythonTextEdits);
    list.sort(Comparator.comparingInt(PythonTextEdit::startLine).thenComparing(PythonTextEdit::startLineOffset));
    Collections.reverse(list);
    return Collections.unmodifiableList(list);
  }

  private static void checkNoCollision(List<PythonTextEdit> pythonTextEdits) throws IllegalArgumentException {
    for (int i = 0; i < pythonTextEdits.size(); i++) {
      PythonTextEdit edit = pythonTextEdits.get(i);
      for (int j = i + 1; j < pythonTextEdits.size(); j++) {
        PythonTextEdit edit2 = pythonTextEdits.get(j);
        if (oneEnclosedByTheOther(edit2, edit)) {
          throw new IllegalArgumentException("There is a collision between the range of the quickfixes.");
        }
      }
    }
  }

  // Returns true if the range of one edit crosses the range of the other. If one end of both edits is the same point,
  // we should return false
  private static boolean oneEnclosedByTheOther(PythonTextEdit toCheck, PythonTextEdit reference) {
    if (onSameLine(toCheck, reference)) {
      // If on same line, we need to check that the bounds of toCheck are not contained in reference bounds
      return !(toCheck.endLineOffset() < reference.startLineOffset() || toCheck.startLineOffset() > reference.endLineOffset());
    } else {
      if (compactOnDifferentLines(toCheck, reference)) {
        return false;
      } else if (isCompact(toCheck)) {
        return isSecondInFirst(toCheck, reference);
      } else if (isCompact(reference)) {
        return isSecondInFirst(reference, toCheck);
      } else {
        // Both edits exploded on different lines
        if (noLineIntersection(toCheck, reference)) {
          return false;
        } else {
          // There is an intersection between edits, only need to check valid case
          if (reference.startLine() == toCheck.endLine()) {
            return !(toCheck.endLineOffset() <= reference.startLineOffset());
          } else if (reference.endLine() == toCheck.startLine()) {
            return !(reference.endLineOffset() <= toCheck.startLineOffset());
          }
        }
      }
    }
    // All other cases are invalid and will cause an intersection
    return true;
  }

  private static boolean onSameLine(PythonTextEdit check, PythonTextEdit ref) {
    return ref.startLine() == ref.endLine() && ref.endLine() == check.startLine() && check.startLine() == check.endLine();
  }

  private static boolean compactOnDifferentLines(PythonTextEdit check, PythonTextEdit ref) {
    return ref.startLine() == ref.endLine() && check.startLine() == check.endLine() && ref.endLine() != check.startLine();
  }

  private static boolean isCompact(PythonTextEdit check) {
    return check.startLine() == check.endLine();
  }

  // Returns true if there is an intersection between the ranges
  // The first parameter is a compact edit, i.e., the edit is only one one line
  private static boolean isSecondInFirst(PythonTextEdit first, PythonTextEdit second) {
    if (first.startLine() == second.startLine()) {
      return second.startLineOffset() < first.endLineOffset();
    } else if (first.endLine() == second.endLine()) {
      return first.startLineOffset() < second.endLineOffset();
    }
    // No intersection
    return false;
  }

  private static boolean noLineIntersection(PythonTextEdit check, PythonTextEdit ref) {
    return check.endLine() < ref.startLine() || check.startLine() > ref.endLine();
  }

  private static int convertPositionToIndex(String fileContent, int line, int lineOffset) {
    int currentLine = 1;
    int currentIndex = 0;
    while (currentLine < line) {
      currentIndex = fileContent.indexOf("\n", currentIndex) + 1;
      currentLine++;
    }
    return currentIndex + lineOffset;
  }

  private static class PythonQuickFixFile implements PythonFile {

    private final String content;

    public PythonQuickFixFile(String content) {
      this.content = content;
    }

    @Override
    public String content() {
      return this.content;
    }

    @Override
    public String fileName() {
      return "PythonQuickFixFile";
    }

    @Override
    public URI uri() {
      return URI.create(this.fileName());
    }

    @Override
    public String key() {
      return "PythonQuickFixFile";
    }
  }
}
