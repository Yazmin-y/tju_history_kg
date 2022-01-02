# -*- coding: utf-8 -*-
from django.shortcuts import render


def index(request):
    context = {}
    return render(request, 'entity_reg.html', context)
