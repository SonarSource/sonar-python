############################################
###                Django                ###
############################################


def django_test():
    from django.conf import settings, mysettings
    settings.configure(DEBUG=True)  # Noncompliant
    settings.configure(DEBUG_PROPAGATE_EXCEPTIONS=True)  # Noncompliant {{Make sure this debug feature is deactivated before delivering the code in production.}}
    #                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    mysettings.configure(DEBUG=True)  # OK
    settings.otherFn(DEBUG=True)  # OK
    settings.configure()

    configure(DEBUG=True) # OK
    configure() # OK
    foo.configure(DEBUG=True) # OK

    def custom_config(config):
        settings.configure(default_settings=config, DEBUG=True)  # Noncompliant

    settings.configure(DEBUG=False) # OK
    settings.configure(OTHER=False) # OK

    DEBUG = True  # OK, filename is not globalsetting.py nor settings.py

def flask_test():
    from flask import Flask
    app = Flask()

    app.debug = True  # Noncompliant

    app.config['DEBUG'] = True  # Noncompliant
    app.config.update({'DEBUG': True})  # FN This should be fixed as part of https://sonarsource.atlassian.net/browse/SONARPY-1541

    app.run(debug=True)  # Noncompliant
    app.run('0.0.0.0', 8080, True)  # Noncompliant

