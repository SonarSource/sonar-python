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
package org.sonar.plugins.python;

import java.io.File;
import java.util.ArrayDeque;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.locks.Lock;
import javax.annotation.CheckForNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.TextRange;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.issue.NewIssue;
import org.sonar.api.batch.sensor.issue.NewIssueLocation;
import org.sonar.api.rule.RuleKey;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;
import org.sonar.plugins.python.indexer.PythonIndexer;

public class IssuesRepository {
  private static final Logger LOG = LoggerFactory.getLogger(IssuesRepository.class);
  private final SensorContext context;
  private final PythonChecks checks;
  private final PythonIndexer indexer;
  private final boolean isInSonarLint;
  private final Lock lock;

  public IssuesRepository(SensorContext context, PythonChecks checks, PythonIndexer indexer, boolean isInSonarLint, Lock lock) {
    this.context = context;
    this.checks = checks;
    this.indexer = indexer;
    this.isInSonarLint = isInSonarLint;
    this.lock = lock;
  }

  public void save(PythonInputFile inputFile, List<PythonCheck.PreciseIssue> issues) {
    issues.forEach(preciseIssue -> save(inputFile, preciseIssue));
  }

  private void save(PythonInputFile inputFile, PythonCheck.PreciseIssue preciseIssue) {
    var ruleKey = checks.ruleKey(preciseIssue.check());
    var newIssue = context
      .newIssue()
      .forRule(ruleKey);

    var cost = preciseIssue.cost();
    if (cost != null) {
      newIssue.gap(cost.doubleValue());
    }

    var primaryLocation = newLocation(inputFile, newIssue, preciseIssue.primaryLocation());
    newIssue.at(primaryLocation);

    var secondaryLocationsFlow = new ArrayDeque<NewIssueLocation>();
    for (var secondaryLocation : preciseIssue.secondaryLocations()) {
      String fileId = secondaryLocation.fileId();
      if (fileId != null) {
        InputFile issueLocationFile = component(fileId, context);
        if (issueLocationFile != null) {
          secondaryLocationsFlow.addFirst(newLocation(new PythonInputFileImpl(issueLocationFile), newIssue, secondaryLocation));
        }
      } else {
        newIssue.addLocation(newLocation(inputFile, newIssue, secondaryLocation));
      }
    }

    // secondary locations on multiple files are only supported using flows
    if (!secondaryLocationsFlow.isEmpty()) {
      secondaryLocationsFlow.addFirst(primaryLocation);
      newIssue.addFlow(secondaryLocationsFlow);
    }

    handleQuickFixes(inputFile.wrappedFile(), ruleKey, newIssue, preciseIssue);

    save(newIssue);
  }

  private void save(NewIssue newIssue) {
    try {
      lock.lock();
      newIssue.save();
    } finally {
      lock.unlock();
    }
  }

  @CheckForNull
  private InputFile component(String fileId, SensorContext sensorContext) {
    var predicate = sensorContext.fileSystem().predicates().is(new File(fileId));
    var inputFile = Optional.ofNullable(sensorContext.fileSystem().inputFile(predicate))
      .orElseGet(() -> indexer.getFileWithId(fileId));
    if (inputFile == null) {
      LOG.debug("Failed to find InputFile for {}", fileId);
    }
    return inputFile;
  }

  private static NewIssueLocation newLocation(PythonInputFile inputFile, NewIssue issue, IssueLocation location) {
    var newLocation = issue.newLocation().on(inputFile.wrappedFile());

    if (location.startLine() != IssueLocation.UNDEFINED_LINE) {
      TextRange range;
      if (location.startLineOffset() == IssueLocation.UNDEFINED_OFFSET) {
        range = inputFile.wrappedFile().selectLine(location.startLine());
      } else {
        range = inputFile.wrappedFile().newRange(location.startLine(), location.startLineOffset(), location.endLine(),
          location.endLineOffset());
      }
      newLocation.at(range);
    }

    var message = location.message();
    if (message != null) {
      newLocation.message(message);
    }
    return newLocation;
  }

  private void handleQuickFixes(InputFile inputFile, RuleKey ruleKey, NewIssue newIssue, PythonCheck.PreciseIssue preciseIssue) {
    if (isInSonarLint) {
      var quickFixes = preciseIssue.quickFixes();
      addQuickFixes(inputFile, ruleKey, quickFixes, newIssue);
    }
  }

  private static void addQuickFixes(InputFile inputFile, RuleKey ruleKey, Iterable<PythonQuickFix> quickFixes, NewIssue sonarLintIssue) {
    try {
      for (var quickFix : quickFixes) {
        var newQuickFix = sonarLintIssue.newQuickFix()
          .message(quickFix.getDescription());

        var edit = newQuickFix.newInputFileEdit().on(inputFile);

        quickFix.getTextEdits().stream()
          .map(pythonTextEdit -> edit.newTextEdit().at(rangeFromTextSpan(inputFile, pythonTextEdit))
            .withNewText(pythonTextEdit.replacementText()))
          .forEach(edit::addTextEdit);
        newQuickFix.addInputFileEdit(edit);
        sonarLintIssue.addQuickFix(newQuickFix);
      }
      // TODO : is this try/catch still necessary ?
    } catch (RuntimeException e) {
      // We still want to report the issue if we did not manage to create a quick fix.
      LOG.warn("Could not report quick fixes for rule: {}. {}: {}", ruleKey, e.getClass().getName(), e.getMessage());
    }
  }

  private static TextRange rangeFromTextSpan(InputFile file, PythonTextEdit pythonTextEdit) {
    return file.newRange(pythonTextEdit.startLine(), pythonTextEdit.startLineOffset(), pythonTextEdit.endLine(),
      pythonTextEdit.endLineOffset());
  }
}
