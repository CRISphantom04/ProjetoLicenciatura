from curses.ascii import SI
from dataclasses import fields
from django import forms
from django.forms import ModelForm
from django_mysql.forms import SimpleListField
from .models import VirtualSpecies
from .models import myuploadfile
from django.core.validators import MaxValueValidator, MinValueValidator


# Form to store the information about the species
class VirtualSpeciesForm(forms.Form):
    model = VirtualSpecies
    
    species_name = forms.CharField(max_length=15)
    birth_rate   = forms.FloatField()
    death_rate   = forms.FloatField()
    spread_rate  = forms.FloatField()
    ticks        = forms.IntegerField()
    delta        = forms.IntegerField()
    limit        = forms.FloatField()
    mean         = forms.CharField(required=False)
    std          = forms.CharField(required=False)
    env_variables= SimpleListField(base_field=forms.CharField())

    #Save the information about the species
    def save(self):
        return VirtualSpecies.objects.create(
            species_name =self.cleaned_data['species_name'],
            birth_rate   =self.cleaned_data['birth_rate'],
            death_rate   =self.cleaned_data['death_rate'],
            spread_rate  =self.cleaned_data['spread_rate'],
            ticks        =self.cleaned_data['ticks'],
            delta        =self.cleaned_data['delta'],
            limit        =self.cleaned_data['limit'],
            mean         =self.cleaned_data['mean'],
            std          =self.cleaned_data['std'],
            env_variables=self.cleaned_data['env_variables']
        )

# Form to store the data of the myuploadfile
class MyUploadFileForm(forms.Form):
    model = myuploadfile

    myfiles = forms.FileField()

    #Save the information of files
    def save(self):
        return myuploadfile.objects.create(
            myfiles =self.cleaned_data['myfiles']
        )

