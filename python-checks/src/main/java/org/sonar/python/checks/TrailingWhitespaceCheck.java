/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2017 SonarSource SA
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
package org.sonar.python.checks;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Grammar;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.python.CharsetAwareVisitor;
import org.sonar.python.PythonCheck;
import org.sonar.squidbridge.SquidAstVisitorContext;
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;

@Rule(
    key = TrailingWhitespaceCheck.CHECK_KEY,
    name = "Lines should not end with trailing whitespaces",
    priority = Priority.MINOR,
    tags = Tags.CONVENTION
)
@SqaleConstantRemediation("1min")
public class TrailingWhitespaceCheck extends PythonCheck implements CharsetAwareVisitor {
  public static final String CHECK_KEY = "S1131";
  public static final String MESSAGE = "Remove the useless trailing whitespaces at the end of this line.";

  private static final Pattern TRAILING_WS = Pattern.compile("\\s$");

  private Charset charset;

  @Override
  public void setCharset(Charset charset) {
    this.charset = charset;
  }

  @Override
  public void visitFile(@Nullable AstNode astNode) {
    final SquidAstVisitorContext<Grammar> context = getContext();
    final File file = context.getFile();
    try (BufferedReader reader = Files.newBufferedReader(file.toPath(), charset)) {
      int lineNr = 0;
      String line;
      while ((line = reader.readLine()) != null) {
        ++lineNr;
        if (TRAILING_WS.matcher(line).find()) {
          addLineIssue(MESSAGE, lineNr);
        }
      }
    } catch (IOException e) {
      throw new IllegalStateException("Could not read " + file, e);
    }
  }

}
