# Generated by Django 4.0.3 on 2022-07-04 15:39

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('projetos', '0014_remove_virtualspecies_mean_remove_virtualspecies_std'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='virtualspecies',
            name='env_variables',
        ),
    ]
