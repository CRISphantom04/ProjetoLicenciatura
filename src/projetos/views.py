from contextlib import nullcontext
from dataclasses import field
from multiprocessing import context
from statistics import mean
from unicodedata import name
from django.shortcuts import render, HttpResponse
from django.core.files.storage import FileSystemStorage
import os
from django.contrib import messages
from django.contrib.auth.decorators import permission_required
import numpy as np
from pathlib import Path
import pandas as pd


from .models import VirtualSpecies
from .models import myuploadfile
from .forms import VirtualSpeciesForm
from .forms import MyUploadFileForm
from SDSim import setup
from SDSim import ocurrenceData

BASE_DIR = Path(__file__).resolve().parent.parent


def home(request):
    return render(request, "projetos/home.html")

#Request for the csv varaible in url
def ocurrence(request):
    csv_show = request.GET.get('path_csv','')
    arr = pd.read_csv(csv_show).values.tolist()
    return render(request, "projetos/ocurrenceData.html", {'arr': arr})

# Create your views here.
# class VirtualSpeciesViews():
#     model = VirtualSpecies
#     form_class = VirtualSpeciesForm
#     template_name = 'projetos/simulation.html'
#     fields = '__all__'


#Request for the data input
def newSpecies(request):
    form = VirtualSpeciesForm()
    path_csv = ""

    if request.method == 'POST':
        form = VirtualSpeciesForm(request.POST)
        if form.is_valid():
            species_name= form.cleaned_data['species_name']
            birth_rate = form.cleaned_data['birth_rate']
            death_rate = form.cleaned_data['death_rate']
            spread_rate = form.cleaned_data['spread_rate']
            ticks = form.cleaned_data['ticks']
            delta = form.cleaned_data['delta']
            limit = form.cleaned_data['limit']

            #Get the environment variables
            env_variables =request.POST.getlist("env_variables")
            env_variables_final = []
            for v in env_variables:
                env_variables_final.append(v.split("/")[1])
           
            #Get the mean and std
            mean  = request.POST.getlist("mean")
            mean = [float(i) for i in mean[0].split(',')]

            std = request.POST.getlist("std")
            std = [float(i) for i in std[0].split(',')]

            n = len(env_variables)  
            path = 'src/media/media'
            user_path = ''
            filename = "media/media/" + species_name + str(ticks-1)
            path_csv = filename + ".csv"  

            VirtualSpecies(species_name=species_name, birth_rate=birth_rate, death_rate=death_rate, spread_rate=spread_rate, ticks=ticks, delta=delta, limit=limit, mean=mean, std=std, env_variables=env_variables_final).save()
            print(form)
            path = os.path.join(BASE_DIR, 'media/media')
            #Function to run the simulation
            setup.setup(species_name, ticks, env_variables_final, mean, std, n, death_rate, birth_rate, spread_rate, path, delta, user_path)
            #Function to csv file
            ocurrenceData.ocurrence_data(species_name, filename, limit)
    
    files = myuploadfile.objects.all()
    context = {'form': form, 'files': files, 'path_csv': path_csv}
    

    return render(request, "projetos/simulation.html", context)


def upload_file(request):
    form = MyUploadFileForm()

    if request.method == 'POST':
        myfile = request.FILES.getlist("file[]")

        for f in myfile:
            myuploadfile(myfiles= f).save()

    return render(request, "projetos/upload.html")

