/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
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
import java.util.Locale;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.plugins.python.dependency.model.Dependencies;
import org.sonar.plugins.python.dependency.model.Dependency;

public class RequirementsTxtParser {

  private static final Logger LOG = LoggerFactory.getLogger(RequirementsTxtParser.class);

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
      if (splittedLine.length >= 1 && splittedLine[0].toUpperCase(Locale.ENGLISH).matches("^([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])$")) {
        dependencySet.add(new Dependency(splittedLine[0]));
      }
    }
    return new Dependencies(dependencySet);
  }
}
