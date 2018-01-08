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
package org.sonar.plugins.python.cpd;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.GenericTokenType;
import com.sonar.sslr.api.Token;
import com.sonar.sslr.api.TokenType;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.cpd.NewCpdTokens;
import org.sonar.python.PythonVisitorContext;
import org.sonar.python.TokenLocation;
import org.sonar.python.api.PythonTokenType;

public class PythonCpdAnalyzer {

  private final SensorContext context;

  public PythonCpdAnalyzer(SensorContext context) {
    this.context = context;
  }

  public void pushCpdTokens(InputFile inputFile, PythonVisitorContext visitorContext) {
    AstNode root = visitorContext.rootTree();
    if (root != null) {
      NewCpdTokens cpdTokens = context.newCpdTokens().onFile(inputFile);
      for (Token token : root.getTokens()) {
        if (!isIgnoredType(token.getType())) {
          TokenLocation location = new TokenLocation(token);
          cpdTokens.addToken(location.startLine(), location.startLineOffset(), location.endLine(), location.endLineOffset(), getImage(token));
        }
      }
      cpdTokens.save();
    }
  }

  private boolean isIgnoredType(TokenType type) {
    return type.equals(PythonTokenType.NEWLINE) ||
      type.equals(PythonTokenType.DEDENT) || // TODO equals { } ???
      type.equals(PythonTokenType.INDENT) ||
      type.equals(GenericTokenType.EOF);
  }

  private String getImage(Token token) {
    return token.getValue();
  }

}
