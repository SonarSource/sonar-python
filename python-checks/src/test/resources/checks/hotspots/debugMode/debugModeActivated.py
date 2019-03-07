############################################
###                Django                ###
############################################

from django.conf import settings

settings.configure(DEBUG=True)  # Noncompliant
settings.configure(DEBUG_PROPAGATE_EXCEPTIONS=True)  # Noncompliant {{Make sure this debug feature is deactivated before delivering the code in production.}}
#                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def custom_config(config):
    settings.configure(default_settings=config, DEBUG=True)  # Noncompliant

settings.configure(DEBUG=False) # OK
settings.configure(OTHER=False) # OK