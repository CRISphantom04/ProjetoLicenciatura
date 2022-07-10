# Generated by Django 4.0.3 on 2022-07-02 18:43

from django.db import migrations, models
import django_mysql.models


class Migration(migrations.Migration):

    dependencies = [
        ('projetos', '0007_remove_myuploadfile_mean_remove_myuploadfile_std'),
    ]

    operations = [
        migrations.AddField(
            model_name='virtualspecies',
            name='mean',
            field=django_mysql.models.ListCharField(models.CharField(max_length=10), default=None, max_length=110, size=10),
        ),
        migrations.AddField(
            model_name='virtualspecies',
            name='std',
            field=django_mysql.models.ListCharField(models.CharField(max_length=10), default=None, max_length=110, size=10),
        ),
    ]
