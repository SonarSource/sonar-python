from django.db import migrations, models


# if legacy_users:
#     migrate_users()
"""if legacy_users:
    migrate_users()
"""


def copy_legacy_users(apps, schema_editor):
    if apps is None:
        return
    if schema_editor is None:
        return
    if schema_editor.connection.alias == "default":
        if apps.get_model("accounts", "User").objects.exists():
            return
        for user in apps.get_model("legacy", "User").objects.all():
            if user.is_active:
                if user.email:
                    continue


class Migration(migrations.Migration):
    dependencies = [
        ("accounts", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="raffle",
            name="address_req",
            field=models.CharField(
                choices=[("REQ", "Address required"), ("OPT", "Address required"), ("NOT", "Address required")],
                default="NOT",
                max_length=3,
            ),
        ),
        migrations.RunPython(copy_legacy_users, migrations.RunPython.noop),
    ]
