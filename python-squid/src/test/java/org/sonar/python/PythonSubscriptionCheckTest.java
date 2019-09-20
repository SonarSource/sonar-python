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

import com.sonar.sslr.api.Token;
import com.sonar.sslr.api.Trivia;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.Test;
import org.sonar.python.PythonCheck.PreciseIssue;
import org.sonar.python.api.tree.FunctionDef;
import org.sonar.python.api.tree.Name;
import org.sonar.python.api.tree.Tree;

import static org.assertj.core.api.Assertions.assertThat;

public class PythonSubscriptionCheckTest {

  private static final File FILE = new File("src/test/resources/file.py");
  public static final String MESSAGE = "message";

  private static List<PreciseIssue> scanFileForIssues(File file, PythonCheck check) {
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(file);
    check.scanFile(context);
    SubscriptionVisitor.analyze(Collections.singletonList((PythonSubscriptionCheck) check), context);
    return context.getIssues();
  }

  @Test
  public void test() {
    TestPythonCheck check = new TestPythonCheck() {
      @Override
      public void initialize(Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
          FunctionDef tree = (FunctionDef) ctx.syntaxNode();
          ctx.addIssue(tree.name(), tree.name().firstToken().value());
        });
      }
    };

    List<PreciseIssue> issues = scanFileForIssues(FILE, check);

    assertThat(issues).hasSize(2);
    PreciseIssue firstIssue = issues.get(0);

    assertThat(firstIssue.cost()).isNull();
    assertThat(firstIssue.secondaryLocations()).isEmpty();

    IssueLocation primaryLocation = firstIssue.primaryLocation();
    assertThat(primaryLocation.message()).isEqualTo("hello");

    assertThat(primaryLocation.startLine()).isEqualTo(1);
    assertThat(primaryLocation.endLine()).isEqualTo(1);
    assertThat(primaryLocation.startLineOffset()).isEqualTo(4);
    assertThat(primaryLocation.endLineOffset()).isEqualTo(9);
  }

  @Test
  public void test_cost() {
    TestPythonCheck check = new TestPythonCheck() {
      @Override
      public void initialize(Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
          FunctionDef pyFunctionDefTree = (FunctionDef) ctx.syntaxNode();
          Name name = pyFunctionDefTree.name();
          ctx.addIssue(name.firstToken(), MESSAGE).withCost(42);
        });
      }
    };

    List<PreciseIssue> issues = scanFileForIssues(FILE, check);
    PreciseIssue firstIssue = issues.get(0);
    assertThat(firstIssue.cost()).isEqualTo(42);
  }

  @Test
  public void test_tokens() {
    TestPythonCheck check = new TestPythonCheck() {
      private List<Token> ignoreList;

      @Override
      public void initialize(Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
          ignoreList = new ArrayList<>();
        });
        context.registerSyntaxNodeConsumer(Tree.Kind.TOKEN, ctx -> {
          PyToken pyToken = (PyToken) ctx.syntaxNode();
          if (ignoreList.contains(pyToken.token())) {
            return;
          }
          ignoreList.add(pyToken.token());
          for (Trivia trivia : pyToken.trivia()) {
            if (trivia.isComment()) {
              ctx.addIssue(new PyTokenImpl(trivia.getToken()), MESSAGE);
            }
          }
        });
      }
    };
    List<PreciseIssue> issues = scanFileForIssues(FILE, check);
    assertThat(issues).hasSize(1);
    assertThat(issues.get(0).primaryLocation().startLine()).isEqualTo(5);
    assertThat(issues.get(0).primaryLocation().endLine()).isEqualTo(5);
  }

  @Test
  public void test_file_line() {
    TestPythonCheck check = new TestPythonCheck() {
      @Override
      public void initialize(Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.PASS_STMT, ctx -> {
          ctx.addLineIssue(MESSAGE, ctx.syntaxNode().firstToken().line());
          ctx.addFileIssue(MESSAGE);
        });
      }
    };

    List<PreciseIssue> issues = scanFileForIssues(FILE, check);
    assertThat(issues).hasSize(2);
    assertThat(issues.get(0).primaryLocation().startLine()).isEqualTo(8);
  }

  private abstract static class TestPythonCheck extends PythonSubscriptionCheck {

  }
}
