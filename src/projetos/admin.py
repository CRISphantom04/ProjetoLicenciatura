from django.contrib import admin

# # Register your models here.
from .models import VirtualSpecies, myuploadfile

admin.site.register(VirtualSpecies)
admin.site.register(myuploadfile)