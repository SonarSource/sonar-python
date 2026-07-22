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
package org.sonar.plugins.python.ruff;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.fs.FileSystem;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.rule.Severity;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.issue.NewExternalIssue;
import org.sonar.api.batch.sensor.issue.NewIssueLocation;
import org.sonar.api.config.Configuration;
import org.sonar.api.rules.RuleType;
import org.sonar.plugins.python.ExternalIssuesSensor;
import org.sonarsource.analyzer.commons.internal.json.simple.parser.ParseException;

import static org.apache.commons.lang3.StringUtils.isEmpty;

public class RuffSensor extends ExternalIssuesSensor {

  private static final Logger LOG = LoggerFactory.getLogger(RuffSensor.class);

  public static final String LINTER_NAME = "Ruff";
  public static final String LINTER_KEY = "ruff";
  public static final String REPORT_PATH_KEY = "sonar.python.ruff.reportPaths";

  private static final Long DEFAULT_CONSTANT_DEBT_MINUTES = 5L;

  @Override
  protected boolean shouldExecute(Configuration conf) {
    return conf.hasKey(REPORT_PATH_KEY);
  }

  @Override
  protected String reportPathKey() {
    return REPORT_PATH_KEY;
  }

  @Override
  protected String linterName() {
    return LINTER_NAME;
  }

  @Override
  protected Logger logger() {
    return LOG;
  }

  @Override
  protected void importReport(File reportPath, SensorContext context, Set<String> unresolvedInputFiles)
    throws IOException, ParseException {
    InputStream in = new FileInputStream(reportPath);
    LOG.info("Importing {}", reportPath);
    // Built lazily and reused across all issues of this report, so the fallback below never re-scans the
    // filesystem per issue.
    RelativePathIndex relativePathIndex = new RelativePathIndex(context.fileSystem());
    RuffJsonReportReader.read(in, issue -> saveIssue(context, issue, unresolvedInputFiles, relativePathIndex));
  }

  private static void saveIssue(SensorContext context, RuffJsonReportReader.Issue issue,
    Set<String> unresolvedInputFiles, RelativePathIndex relativePathIndex) {
    if (isEmpty(issue.ruleKey) || isEmpty(issue.filePath) || isEmpty(issue.message)) {
      LOG.debug("Missing information for ruleKey:'{}', filePath:'{}', message:'{}'", issue.ruleKey, issue.filePath,
        issue.message);
      return;
    }

    InputFile inputFile = findInputFile(context.fileSystem(), issue.filePath, relativePathIndex);
    if (inputFile == null) {
      unresolvedInputFiles.add(issue.filePath);
      return;
    }

    NewExternalIssue newExternalIssue = context.newExternalIssue();
    newExternalIssue
      .type(RuleType.CODE_SMELL)
      .severity(Severity.MAJOR)
      .remediationEffortMinutes(DEFAULT_CONSTANT_DEBT_MINUTES);

    NewIssueLocation primaryLocation = newExternalIssue.newLocation()
      .message(issue.message)
      .on(inputFile);

    if (issue.startLocationRow != null) {
      if (isValidEndLocation(issue, inputFile)) {
        primaryLocation.at(inputFile.newRange(issue.startLocationRow, issue.startLocationCol, issue.endLocationRow,
          issue.endLocationCol));
      } else {
        primaryLocation.at(inputFile.selectLine(issue.startLocationRow));
      }
    }

    newExternalIssue.at(primaryLocation);
    newExternalIssue.engineId(LINTER_KEY);
    newExternalIssue.ruleId(issue.ruleKey).save();
  }

  /**
   * Ruff only reports absolute file paths, which may have been computed with a different base directory than the
   * one used for the current analysis (e.g. report generated in CI, analysis run locally, or vice versa).
   * When the direct lookup fails, fall back to matching the report path against the relative path of an indexed
   * file, disregarding the base directory, through the {@link RelativePathIndex}.
   */
  private static InputFile findInputFile(FileSystem fileSystem, String filePath, RelativePathIndex relativePathIndex) {
    InputFile inputFile = fileSystem.inputFile(fileSystem.predicates().hasPath(filePath));
    if (inputFile != null) {
      return inputFile;
    }
    return relativePathIndex.findByRelativePathSuffix(filePath);
  }

  /**
   * Indexes input files by file name so that the suffix-matching fallback only has to compare the (typically few)
   * files sharing the same name, instead of scanning every indexed file for every unresolved issue. Built lazily,
   * on first use, and reused for the whole report.
   * <p>
   * Among several candidates sharing the report path's file name, the one whose relative path is the longest
   * matching suffix is preferred, since it pins down the project base directory boundary more precisely than a
   * shorter, coincidental match (e.g. a root-level {@code __init__.py} trivially suffix-matches every deeper
   * {@code __init__.py} report path). Candidates matching an equally long suffix are rejected as ambiguous.
   * <p>
   * This remains a best-effort heuristic: if the file Ruff actually analyzed isn't indexed at all, the issue can
   * still be attached to an unrelated file that happens to share its relative path suffix.
   */
  private static class RelativePathIndex {

    private final FileSystem fileSystem;
    private Map<String, List<InputFile>> filesByName;

    private RelativePathIndex(FileSystem fileSystem) {
      this.fileSystem = fileSystem;
    }

    InputFile findByRelativePathSuffix(String filePath) {
      if (filesByName == null) {
        filesByName = buildIndex(fileSystem);
      }
      String normalizedFilePath = filePath.replace('\\', '/');
      List<InputFile> candidates = filesByName.getOrDefault(fileName(normalizedFilePath), List.of());

      InputFile bestMatch = null;
      int bestMatchLength = -1;
      boolean ambiguous = false;
      for (InputFile candidate : candidates) {
        String relativePath = candidate.relativePath();
        if (normalizedFilePath.equals(relativePath) || normalizedFilePath.endsWith("/" + relativePath)) {
          if (relativePath.length() > bestMatchLength) {
            bestMatch = candidate;
            bestMatchLength = relativePath.length();
            ambiguous = false;
          } else if (relativePath.length() == bestMatchLength) {
            ambiguous = true;
          }
        }
      }
      if (ambiguous) {
        LOG.debug("Multiple files equally match the path '{}', skipping ambiguous match", filePath);
        return null;
      }
      if (bestMatch != null) {
        LOG.debug("Resolved report path '{}' to '{}' by matching the relative path, ignoring the base directory", filePath, bestMatch);
      }
      return bestMatch;
    }

    private static Map<String, List<InputFile>> buildIndex(FileSystem fileSystem) {
      Map<String, List<InputFile>> index = new HashMap<>();
      for (InputFile inputFile : fileSystem.inputFiles(fileSystem.predicates().all())) {
        index.computeIfAbsent(fileName(inputFile.relativePath()), k -> new ArrayList<>()).add(inputFile);
      }
      return index;
    }

    private static String fileName(String path) {
      return path.substring(path.lastIndexOf('/') + 1);
    }
  }

  /*
   * The end location column should be after the start location col
   */
  private static boolean isValidEndLocation(RuffJsonReportReader.Issue issue, InputFile inputFile) {
    return issue.startLocationCol != null &&
      issue.endLocationRow != null &&
      issue.endLocationCol != null &&
      isColInBounds(issue.startLocationRow, issue.startLocationCol, inputFile) &&
      ((issue.endLocationRow.equals(issue.startLocationRow) && issue.endLocationCol > issue.startLocationCol) ||
        !issue.endLocationRow.equals(issue.startLocationRow));

  }

  private static boolean isColInBounds(int lineNumber, int columnNumber, InputFile inputFile) {
    return columnNumber < inputFile.selectLine(lineNumber).end().lineOffset();
  }

}
