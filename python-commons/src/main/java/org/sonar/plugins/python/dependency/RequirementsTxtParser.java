/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.fs.FilePredicates;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.plugins.python.dependency.model.Dependencies;
import org.sonar.plugins.python.dependency.model.Dependency;

public class RequirementsTxtParser {

  private static final Logger LOG = LoggerFactory.getLogger(RequirementsTxtParser.class);

  private RequirementsTxtParser(){}

  public static Dependencies parseRequirementFiles(SensorContext context) {
    List<InputFile> requirementFiles = getRequirementFiles(context);

    Set<Dependency> dependencies = new HashSet<>();
    for (InputFile requirementFile : requirementFiles) {
      dependencies.addAll(parseRequirementFile(requirementFile));
    }
    return new Dependencies(dependencies);
  }

  private static Set<Dependency> parseRequirementFile(InputFile requirementFile) {
    Set<Dependency> dependencySet = new HashSet<>();
    String fileContent = "";
    try {
      fileContent = requirementFile.contents();
    } catch (IOException e) {
      LOG.warn("There was an exception when parsing {}. No dependencies were extracted.", requirementFile.filename(), e);
      return new HashSet<>();
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
    return dependencySet;
  }

  static List<InputFile> getRequirementFiles(SensorContext context) {
    FilePredicates p = context.fileSystem().predicates();
    Iterable<InputFile> it = context.fileSystem().inputFiles(p.and(p.hasFilename("requirements.txt")));
    List<InputFile> list = new ArrayList<>();
    it.forEach(list::add);
    return Collections.unmodifiableList(list);
  }

}
