from django.contrib import admin
from django.urls import path, include
from .views import iwantvid, pred
from django.conf import settings
from django.conf.urls.static import static



urlpatterns = [
    path('api/', iwantvid, name='just-a-video'),
    path('predict/',pred, name='predict_video'),
]

