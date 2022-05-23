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

import com.sonar.sslr.api.TokenType;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.api.batch.fs.TextRange;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.tree.Trivia;
import org.sonar.python.TokenLocation;
import org.sonar.python.tree.TokenImpl;
import org.sonar.python.tree.TriviaImpl;

public abstract class IssueLocation {

  public static final int UNDEFINED_OFFSET = -1;

  public static final int UNDEFINED_LINE = 0;

  private String message;

  private IssueLocation(@Nullable String message) {
    this.message = message;
  }

  public static IssueLocation atFileLevel(String message) {
    return new FileLevelIssueLocation(message);
  }

  public static IssueLocation atLineLevel(String message, int lineNumber) {
    return new LineLevelIssueLocation(message, lineNumber);
  }

  public static IssueLocation preciseLocation(Tree tree, @Nullable String message) {
    return new PreciseIssueLocation(tree.firstToken(), tree.lastToken(), message);
  }

  public static IssueLocation preciseLocation(Token token, @Nullable String message) {
    return new PreciseIssueLocation(token, message);
  }

  public static IssueLocation preciseLocation(Token from, Token to, @Nullable String message) {
    return new PreciseIssueLocation(from, to, message);
  }

  public static IssueLocation preciseLocation(LocationInFile locationInFile, @Nullable String message) {
    return new PreciseIssueLocation(locationInFile, message);
  }

  @CheckForNull
  public String message() {
    return message;
  }

  public abstract int startLine();

  public abstract int startLineOffset();

  public abstract int endLine();

  public abstract int endLineOffset();

  @CheckForNull
  public abstract String fileId();

  private static class PreciseIssueLocation extends IssueLocation {
    @CheckForNull
    private final String fileId;
    private final int startLine;
    private final int startLineOffset;
    private final int endLine;
    private final int endLineOffset;

    public PreciseIssueLocation(Token firstToken, Token lastToken, @Nullable String message) {
      super(message);
      startLine = firstToken.line();
      startLineOffset = firstToken.column();
      TokenLocation tokenLocation = new TokenLocation(lastToken);
      endLine = tokenLocation.endLine();
      endLineOffset = tokenLocation.endLineOffset();
      fileId = null;
    }

    public PreciseIssueLocation(Token token, @Nullable String message) {
      this(token, token, message);
    }

    public PreciseIssueLocation(LocationInFile locationInFile, @Nullable String message) {
      super(message);
      startLine = locationInFile.startLine();
      startLineOffset = locationInFile.startLineOffset();
      endLine = locationInFile.endLine();
      endLineOffset = locationInFile.endLineOffset();
      fileId = locationInFile.fileId();
    }

    @Override
    public int startLine() {
      return startLine;
    }

    @Override
    public int startLineOffset() {
      return startLineOffset;
    }

    @Override
    public int endLine() {
      return endLine;
    }

    @Override
    public int endLineOffset() {
      return endLineOffset;
    }

    @Override
    public String fileId() {
      return fileId;
    }

  }

  private static class LineLevelIssueLocation extends IssueLocation {

    private final int lineNumber;

    public LineLevelIssueLocation(String message, int lineNumber) {
      super(message);
      this.lineNumber = lineNumber;
    }

    @Override
    public int startLine() {
      return lineNumber;
    }

    @Override
    public int startLineOffset() {
      return UNDEFINED_OFFSET;
    }

    @Override
    public int endLine() {
      return lineNumber;
    }

    @Override
    public int endLineOffset() {
      return UNDEFINED_OFFSET;
    }

    @Override
    public String fileId() {
      return null;
    }

  }

  private static class FileLevelIssueLocation extends IssueLocation {

    public FileLevelIssueLocation(String message) {
      super(message);
    }

    @Override
    public int startLine() {
      return UNDEFINED_LINE;
    }

    @Override
    public int startLineOffset() {
      return UNDEFINED_OFFSET;
    }

    @Override
    public int endLine() {
      return UNDEFINED_LINE;
    }

    @Override
    public int endLineOffset() {
      return UNDEFINED_OFFSET;
    }

    @Override
    public String fileId() {
      return null;
    }

  }

  public static class PythonTextEdit extends PreciseIssueLocation{

    public PythonTextEdit(Token onlyToken, @Nullable String message){
      super(onlyToken, message);
    }

    public PythonTextEdit(PreciseIssueLocation preciseIssueLocation, String text){
      super(PythonTextEdit.fromPreciseIssue(preciseIssueLocation), text);
    }
    private static LocationInFile fromPreciseIssue(PreciseIssueLocation issueLocation){
      return new LocationInFile(issueLocation.fileId, issueLocation.startLine, issueLocation.startLineOffset, issueLocation.endLine, issueLocation.endLineOffset);
    }


    public PythonTextEdit(LocationInFile location, String addition){
      super(location, addition);
    }

    public static PythonTextEdit insertAtPosition(IssueLocation issueLocation, String addition){
      LocationInFile location = atBeginningOfIssue((PreciseIssueLocation) issueLocation);
      return new PythonTextEdit(location, addition);
    }
    private static LocationInFile atBeginningOfIssue(PreciseIssueLocation issue){
      return new LocationInFile(issue.fileId, issue.startLine, issue.startLineOffset, issue.startLine, issue.startLineOffset);
    }


    public void insertAfter(Token token, String text){}

    public void insertAfterRange(TextRange textRange, String text){}

//    public PythonTextEdit insertBefore(Tree tree, String addition){
////      Token firstToken = tree.firstToken();
////      if (firstToken == null) {
////        throw new IllegalStateException("Trying to insert a quick fix before a Tree without token.");
////      }
////      return insertAtPosition(firstToken.line(), firstToken.column(), addition);
//    }


    public PythonTextEdit insertBefore(Token token, String text){
//      return new PythonTextEdit(token, text);
      Token newToken = new Token() {
        @Override
        public void accept(TreeVisitor visitor) {
          token.accept(visitor);
        }

        @Override
        public boolean is(Kind... kinds) {
          return token.is(kinds);
        }

        @Override
        public Token firstToken() {
          return token.firstToken();
        }

        @Override
        public Token lastToken() {
          return token.lastToken();
        }

        @Override
        public Tree parent() {
          //TODO
          return token.parent();
        }

        @Override
        public List<Tree> children() {
          //TODO
          return token.children();
        }

        @Override
        public Kind getKind() {
          //TODO
          return token.getKind();
        }

        @Override
        public String value() {
          return text + token.value();
        }

        @Override
        public int line() {
          return token.line();
        }

        @Override
        public int column() {
          return token.column() - text.length();
        }

        @Override
        public List<Trivia> trivia() {
          List<Trivia> trivias = new ArrayList<>();
          com.sonar.sslr.api.Token tokenToAdd =com.sonar.sslr.api.Token.builder()
            .setValueAndOriginalValue(text)
            .setColumn(token.column()-text.length())
            .setLine(token.line())
            .build();
          trivias.add(new TriviaImpl(new TokenImpl(tokenToAdd)));
          trivias.addAll(token.trivia());
          return trivias;
        }

        @Override
        public TokenType type() {
          return token.type();
        }
      };
      return new PythonTextEdit(newToken, message());
    }

    public void insertBeforeRange(TextRange textRange, String text){}

    public void remove(Token token){}

    public void removeRange(TextRange textRange){}

    public void replace(Token token, String text){}

    public void replaceRange(TextRange textRange, String text) {}

  }
}
