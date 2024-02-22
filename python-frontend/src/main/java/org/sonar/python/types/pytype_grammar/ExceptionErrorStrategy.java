/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python.types.pytype_grammar;

import org.antlr.v4.runtime.DefaultErrorStrategy;
import org.antlr.v4.runtime.InputMismatchException;
import org.antlr.v4.runtime.Parser;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.misc.IntervalSet;

public class ExceptionErrorStrategy extends DefaultErrorStrategy {

  @Override
  public void recover(Parser recognizer, RecognitionException e) {
    throw e;
  }

  @Override
  public void reportInputMismatch(Parser recognizer, InputMismatchException e) throws RecognitionException {
    String msg = "mismatched input " + getTokenErrorDisplay(e.getOffendingToken());
    msg += " expecting one of " + e.getExpectedTokens().toString(recognizer.getTokenNames());
    RecognitionException ex = new RecognitionException(msg, recognizer, recognizer.getInputStream(), recognizer.getContext());
    ex.initCause(e);
    throw ex;
  }

  @Override
  public void reportMissingToken(Parser recognizer) {
    beginErrorCondition(recognizer);
    Token t = recognizer.getCurrentToken();
    IntervalSet expecting = getExpectedTokens(recognizer);
    String msg = "missing " + expecting.toString(recognizer.getTokenNames()) + " at " + getTokenErrorDisplay(t);
    throw new RecognitionException(msg, recognizer, recognizer.getInputStream(), recognizer.getContext());
  }

  @Override
  protected void reportUnwantedToken(Parser recognizer) {
    this.beginErrorCondition(recognizer);
    Token t = recognizer.getCurrentToken();
    String tokenName = this.getTokenErrorDisplay(t);
    IntervalSet expecting = this.getExpectedTokens(recognizer);
    String msg = "extraneous input " + tokenName + " expecting " + expecting.toString(recognizer.getVocabulary());
    throw new RecognitionException(msg, recognizer, recognizer.getInputStream(), recognizer.getContext());
  }
}
