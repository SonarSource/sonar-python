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
package org.sonar.plugins.python.coverage;

import java.io.File;
import java.util.Map;
import javax.xml.stream.XMLStreamException;
import org.apache.commons.lang.StringUtils;
import org.codehaus.staxmate.in.SMInputCursor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.coverage.NewCoverage;
import org.sonar.plugins.python.EmptyReportException;
import org.sonar.plugins.python.parser.StaxParser;

public class CoberturaParser {

  private static final Logger LOG = LoggerFactory.getLogger(CoberturaParser.class);

  private int unresolvedFilenameCount;

  public void parseReport(File xmlFile, SensorContext context, final Map<InputFile, NewCoverage> coverageData) throws XMLStreamException {
    LOG.info("Parsing report '{}'", xmlFile);
    unresolvedFilenameCount = 0;

    StaxParser parser = new StaxParser(rootCursor -> {
      File baseDirectory = context.fileSystem().baseDir();
      try {
        rootCursor.advance();
      } catch (com.ctc.wstx.exc.WstxEOFException eofExc) {
        LOG.debug("Unexpected end of file is encountered", eofExc);
        throw new EmptyReportException();
      }
      SMInputCursor cursor = rootCursor.childElementCursor();
      while (cursor.getNext() != null) {
        if ("sources".equals(cursor.getLocalName())) {
          baseDirectory = extractBaseDirectory(cursor, baseDirectory);
        } else if ("packages".equals(cursor.getLocalName())) {
          collectFileMeasures(cursor.descendantElementCursor("class"), context, coverageData, baseDirectory);
        }
      }
    });
    parser.parse(xmlFile);
    if (unresolvedFilenameCount > 1) {
      LOG.error("Cannot resolve {} file paths, ignoring coverage measures for those files", unresolvedFilenameCount);
    }
  }

  private static File extractBaseDirectory(SMInputCursor sources, File defaultBaseDirectory) throws XMLStreamException {
    SMInputCursor source = sources.childElementCursor("source");
    while (source.getNext() != null) {
      String path = source.collectDescendantText();
      if (!StringUtils.isBlank(path)) {
        File baseDirectory = new File(path);
        if (baseDirectory.isDirectory()) {
          return baseDirectory;
        } else {
          LOG.warn("Invalid directory path in 'source' element: {}", path);
        }
      }
    }
    return defaultBaseDirectory;
  }

  private void collectFileMeasures(SMInputCursor classCursor, SensorContext context, Map<InputFile, NewCoverage> coverageData, File baseDirectory)
    throws XMLStreamException {
    while (classCursor.getNext() != null) {
      File file = new File(classCursor.getAttrValue("filename"));
      if (!file.isAbsolute()) {
        file = new File(baseDirectory, file.getPath());
      }
      InputFile inputFile = context.fileSystem().inputFile(context.fileSystem().predicates().hasAbsolutePath(file.getAbsolutePath()));
      if (inputFile != null) {
        NewCoverage coverage = coverageData.computeIfAbsent(inputFile, f -> context.newCoverage().onFile(f));
        collectFileData(classCursor, coverage);
      } else {
        classCursor.advance();
        unresolvedFilenameCount++;
        if (unresolvedFilenameCount == 1) {
          LOG.error("Cannot find the file '{}' in the base directory '{}', ignoring coverage measures for this file", file.getPath(), baseDirectory.getPath());
        }
      }
    }
  }

  private static void collectFileData(SMInputCursor classCursor, NewCoverage coverage) throws XMLStreamException {
    SMInputCursor line = classCursor.childElementCursor("lines").advance().childElementCursor("line");
    while (line.getNext() != null) {
      int lineId = Integer.parseInt(line.getAttrValue("number"));
      coverage.lineHits(lineId, Integer.parseInt(line.getAttrValue("hits")));

      String isBranch = line.getAttrValue("branch");
      String text = line.getAttrValue("condition-coverage");
      if (StringUtils.equals(isBranch, "true") && StringUtils.isNotBlank(text)) {
        String[] conditions = StringUtils.split(StringUtils.substringBetween(text, "(", ")"), "/");
        coverage.conditions(lineId, Integer.parseInt(conditions[1]), Integer.parseInt(conditions[0]));
      }
    }
  }
}
