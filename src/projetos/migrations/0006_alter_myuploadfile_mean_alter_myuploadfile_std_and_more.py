# Generated by Django 4.0.3 on 2022-06-30 11:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('projetos', '0005_alter_myuploadfile_mean_alter_myuploadfile_std'),
    ]

    operations = [
        migrations.AlterField(
            model_name='myuploadfile',
            name='mean',
            field=models.FloatField(),
        ),
        migrations.AlterField(
            model_name='myuploadfile',
            name='std',
            field=models.FloatField(),
        ),
        migrations.AlterField(
            model_name='virtualspecies',
            name='birth_rate',
            field=models.FloatField(),
        ),
        migrations.AlterField(
            model_name='virtualspecies',
            name='death_rate',
            field=models.FloatField(),
        ),
        migrations.AlterField(
            model_name='virtualspecies',
            name='limit',
            field=models.FloatField(),
        ),
        migrations.AlterField(
            model_name='virtualspecies',
            name='spread_rate',
            field=models.FloatField(),
        ),
    ]