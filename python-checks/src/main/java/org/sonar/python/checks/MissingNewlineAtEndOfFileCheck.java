/*
 * SonarQube Python Plugin
 * Copyright (C) 2011 SonarSource and Waleri Enns
 * dev@sonar.codehaus.org
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */
package org.sonar.python.checks;

import com.google.common.io.Files;
import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Grammar;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.python.CharsetAwareVisitor;
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;
import org.sonar.squidbridge.annotations.SqaleSubCharacteristic;
import org.sonar.squidbridge.checks.SquidCheck;

import javax.annotation.Nullable;
import java.io.IOException;
import java.nio.charset.Charset;

@Rule(
    key = MissingNewlineAtEndOfFileCheck.CHECK_KEY,
    priority = Priority.MINOR,
    name = "Files should contain an empty new line at the end",
    tags = Tags.CONVENTION
)
@SqaleSubCharacteristic(RulesDefinition.SubCharacteristics.READABILITY)
@SqaleConstantRemediation("1min")
public class MissingNewlineAtEndOfFileCheck extends SquidCheck<Grammar> implements CharsetAwareVisitor {
  public static final String CHECK_KEY = "S113";
  public static final String MESSAGE = "Add a new line at the end of this file \"%s\".";
  private Charset charset;

  @Override
  public void setCharset(Charset charset) {
    this.charset = charset;
  }

  @Override
  public void visitFile(@Nullable AstNode astNode) {
    String[] lines;
    try {
      String fileContent = Files.toString(getContext().getFile(), charset);
      lines = fileContent.split("[\r\n]", -1);
    } catch (IOException e) {
      throw new IllegalStateException("Could not read " + getContext().getFile(), e);
    }
    if (!"".equals(lines[lines.length - 1])) {
      getContext().createFileViolation(this, String.format(MESSAGE, getContext().getFile().getName()));

    }
  }

}
