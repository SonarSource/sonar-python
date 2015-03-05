#!/usr/bin/env python
# -*- mode: python; coding: iso-8859-1 -*-

# SonarQube Python Plugin
# Copyright (C) Waleri Enns
# dev@sonar.codehaus.org

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02


import os
import re
import json
import requests
from requests.auth import HTTPBasicAuth
import subprocess
from behave import given, when, then, model
from common import analyselog

SONAR_URL = "http://localhost:9000"
TESTDATADIR = os.path.normpath(os.path.join(os.path.realpath(__file__),
                                            "..", "..", "..", "testdata"))
DEFAULT_PROFILE = "Sonar way"
PYTHON = "py"
BASIC_AUTH=HTTPBasicAuth('admin', 'admin')

@given(u'the python project "{project}"')
def step_impl(context, project):
    assert os.path.isdir(os.path.join(TESTDATADIR, project))
    context.project = project
    context.qprofile_key = None


@when(u'I run "{command}"')
def step_impl(context, command):
    _run_command(context, command)


@then(u'the analysis finishes successfully')
def step_impl(context):
    assert context.rc == 0, "Exit code is %i, but should be zero" % context.rc


@then(u'the analysis log contains no error or warning messages')
def step_impl(context):
    badlines, _errors, _warnings = analyselog(context.log)

    assert len(badlines) == 0,\
        ("Found following errors and/or warnings lines in the logfile:\n"
         + "".join(badlines)
         + "For details see %s" % context.log)


@then(u'the following metrics have following values')
def step_impl(context):
    def _toSimpleDict(measures):
        if isinstance(measures, model.Table):
            return {row["metric"]: None if row["value"] == "None" else float(row["value"])
                    for row in measures}
        else:
            return {measure["key"]: measure["val"] for measure in measures}

    exp_measures = _toSimpleDict(context.table)
    metrics_to_query = exp_measures.keys()

    try:
        url = (SONAR_URL + "/api/resources?resource=" + context.project + "&metrics="
               + ",".join(metrics_to_query))
        response = requests.get(url)
        got_measures = {}
        json_measures = json.loads(response.text)[0].get("msr", None)
        if json_measures is not None:
            got_measures = _toSimpleDict(json_measures)
        diff = _diffMeasures(exp_measures, got_measures)
    except requests.exceptions.ConnectionError, e:
        assert False, "cannot query the metrics, details: %s" % str(e)

    assert diff == "", "\n" + diff


@then(u'the analysis breaks')
def step_impl(context):
    assert context.rc != 0, "Exit code is %i, but should be non zero" % context.rc


@then(u'the analysis log contains a line matching')
def step_impl(context):
    pattern = re.compile(context.text)
    with open(context.log) as logfo:
        for line in logfo:
            if pattern.match(line):
                return
    assert False


@when(u'I run sonar-runner with following options')
def step_impl(context):
    arguments = [line for line in context.text.split("\n") if line != '']
    command = "sonar-runner " + " ".join(arguments)
    _run_command(context, command)


@then(u'the number of violations fed is {number}')
def step_impl(context, number):
    exp_measures = {"violations": float(number)}
    assert_measures(context.project, exp_measures)


@given(u'only Pylint rules are active')
def step_impl(context):
    _deactivate_all_rules(context)
    _activate_pylint_rules(context)
    context.cleanup_callbacks.append(_reset_default_profile)


def _activate_pylint_rules(context):
    if context.qprofile_key is None:
        context.qprofile_key = _default_profile_key()
    url = SONAR_URL + "/api/qualityprofiles/activate_rules"
    payload = {'profile_key': context.qprofile_key, 'repositories': 'Pylint'}
    response = requests.post(url, payload, auth=BASIC_AUTH)
    if response.status_code != requests.codes.ok:
        assert False, "cannot activate Pylint rules, error code: %s, message: '%s'" % (response.status_code, response.text)


def _deactivate_all_rules(context):
    if context.qprofile_key is None:
        context.qprofile_key = _default_profile_key()
    url = SONAR_URL + "/api/qualityprofiles/deactivate_rules"
    payload = {'profile_key': context.qprofile_key}
    response = requests.post(url, payload, auth=BASIC_AUTH)
    if response.status_code != requests.codes.ok:
        assert False, "cannot deactivate rules, error code: %s, message: '%s'" % (response.status_code, response.text)


def _reset_default_profile(context):
    url = SONAR_URL + "/api/qualityprofiles/restore_built_in"
    payload = {"language": "py"}
    response = requests.post(url, payload, auth=BASIC_AUTH)


def assert_measures(project, measures):
    metrics_to_query = measures.keys()

    try:
        url = (SONAR_URL + "/api/resources?resource=" + project + "&metrics="
               + ",".join(metrics_to_query))
        response = requests.get(url)
        got_measures = {}
        json_measures = json.loads(response.text)[0].get("msr", None)
        if json_measures is not None:
            got_measures = _gotMeasuresToDict(json_measures)

        diff = _diffMeasures(measures, got_measures)
    except requests.exceptions.ConnectionError, e:
        assert False, "cannot query the metrics, details: %s" % str(e)

    assert diff == "", "\n" + diff


def _diffMeasures(expected, measured):
    difflist = []
    for metric, value_expected in expected.iteritems():
        value_measured = measured.get(metric, None)
        if value_expected != value_measured:
            difflist.append("\t%s is actually %s" % (metric, str(value_measured)))
    return "\n".join(difflist)


def _gotMeasuresToDict(measures):
    return {measure["key"]: measure["val"] for measure in measures}


def _default_profile_key():
    response = requests.get(SONAR_URL + "/api/rules/app")
    qprofiles = json.loads(response.text).get("qualityprofiles", None)
    profiledict = {measure["name"] + " - " + measure["lang"]: measure["key"]
                   for measure in qprofiles}
    return profiledict["%s - %s" % (DEFAULT_PROFILE, PYTHON)]


def _run_command(context, command):
    context.log = "_%s_.log" % context.project
    projecthome = os.path.join(TESTDATADIR, context.project)
    with open(context.log, "w") as logfile:
        rc = subprocess.call(command,
                             cwd=projecthome,
                             stdout=logfile, stderr=logfile,
                             shell=True)
    context.rc = rc
