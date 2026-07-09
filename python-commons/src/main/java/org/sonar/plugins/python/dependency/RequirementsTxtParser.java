/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python.dependency;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.plugins.python.dependency.model.Dependencies;
import org.sonar.plugins.python.dependency.model.Dependency;

public class RequirementsTxtParser {

  private static final Logger LOG = LoggerFactory.getLogger(RequirementsTxtParser.class);

  // Matches a PEP 508 package name, stopping before extras ("[...]") or version specifiers/markers glued without a space
  private static final Pattern PACKAGE_NAME_PATTERN =
    Pattern.compile("^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?(?=$|[\\[<>=!~;@])");

  private RequirementsTxtParser(){}

  public static Dependencies parseRequirementFile(InputFile requirementFile) {
    Set<Dependency> dependencySet = new HashSet<>();
    String fileContent = "";
    try {
      fileContent = requirementFile.contents();
    } catch (IOException e) {
      LOG.warn("There was an exception when parsing {}. No dependencies were extracted.", requirementFile.filename(), e);
      return new Dependencies(Set.of());
    }
    for (String line : fileContent.lines().toList()) {
      line = line.strip();
      if (line.isEmpty() || line.startsWith("#") || line.startsWith("-")) {
        continue;
      }
      String[] splittedLine = line.split("\\s+");
      if (splittedLine.length >= 1) {
        Matcher matcher = PACKAGE_NAME_PATTERN.matcher(splittedLine[0]);
        if (matcher.find()) {
          dependencySet.add(new Dependency(matcher.group()));
        }
      }
    }
    return new Dependencies(dependencySet);
  }
}
