from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db.models import CheckConstraint, Q
from numpy import size
from django_mysql.models import ListCharField



# Model to store the information about the species
class VirtualSpecies(models.Model):
    id            = models.AutoField(primary_key=True)
    species_name  = models.CharField(max_length=15)
    birth_rate    = models.FloatField()
    death_rate    = models.FloatField()
    spread_rate   = models.FloatField()
    ticks         = models.IntegerField()
    delta         = models.IntegerField()
    limit         = models.FloatField()
    mean          = ListCharField(base_field=models.CharField(max_length=10), size=10, max_length=(10*11), default=None)
    std           = ListCharField(base_field=models.CharField(max_length=10), size=10, max_length=(10*11), default=None)
    env_variables = ListCharField(base_field=models.CharField(max_length=20), size=10, max_length=(20*11), default=None)


    def __str__(self):
        return self.species_name



#Model to store the data of the myuploadfile
class myuploadfile(models.Model):
    id      = models.AutoField(primary_key=True)
    myfiles = models.FileField(upload_to='media/')
    
