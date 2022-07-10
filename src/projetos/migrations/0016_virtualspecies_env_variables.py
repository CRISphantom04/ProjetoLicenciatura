# Generated by Django 4.0.3 on 2022-07-04 15:55

from django.db import migrations, models
import django_mysql.models


class Migration(migrations.Migration):

    dependencies = [
        ('projetos', '0015_remove_virtualspecies_env_variables'),
    ]

    operations = [
        migrations.AddField(
            model_name='virtualspecies',
            name='env_variables',
            field=django_mysql.models.ListCharField(models.CharField(max_length=20), default=None, max_length=220, size=10),
        ),
    ]
