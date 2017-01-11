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

  public void parseReport(File xmlFile, SensorContext context, final Map<InputFile, NewCoverage> coverageData) throws XMLStreamException {
    LOG.info("Parsing report '{}'", xmlFile);

    StaxParser parser = new StaxParser(rootCursor -> {
      try {
        rootCursor.advance();
      } catch (com.ctc.wstx.exc.WstxEOFException eofExc) {
        LOG.debug("Unexpected end of file is encountered", eofExc);
        throw new EmptyReportException();
      }
      collectPackageMeasures(rootCursor.descendantElementCursor("package"), context, coverageData);
    });
    parser.parse(xmlFile);
  }

  private static void collectPackageMeasures(SMInputCursor pack, SensorContext context, Map<InputFile, NewCoverage> coverageData) throws XMLStreamException {
    while (pack.getNext() != null) {
      collectFileMeasures(pack.descendantElementCursor("class"), context, coverageData);
    }
  }

  private static void collectFileMeasures(SMInputCursor classCursor, SensorContext context, Map<InputFile, NewCoverage> coverageData) throws XMLStreamException {
    while (classCursor.getNext() != null) {
      String fileName = classCursor.getAttrValue("filename");
      InputFile inputFile = context.fileSystem().inputFile(context.fileSystem().predicates().hasPath(fileName));

      if (inputFile != null) {
        NewCoverage coverage = coverageData.get(inputFile);
        if (coverage == null) {
          coverage = context.newCoverage().onFile(inputFile);
          coverageData.put(inputFile, coverage);
        }
        collectFileData(classCursor, coverage);

      } else {
        LOG.debug("Cannot find the file '{}', ignoring coverage measures", fileName);
        classCursor.getNext();
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
