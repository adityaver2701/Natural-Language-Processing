# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 11:38:45 2020

@author: kavya
"""

import json
import traceback

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from core.lib.lib import pos_tagger, ner_tagger, parser, sentiment, tags, parser_svg
from core.constants import *
from core.lib.text_generation import text_generator
@csrf_exempt
def text_generation_controller(request):
    try:
        if request.method == "POST":
            print('inside controller')
            params = json.loads(request.body.decode(CHARSET))
            print('outside json loads')
            print(request.body)
            print(params)
            print(type(params))
            response = text_generator(params['body'])
            
            return HttpResponse(json.dumps(response), content_type=CONTENT_TYPE)
    except Exception as e:
        traceback.print_exc()
        return HttpResponse(e.args, status=500, content_type=CONTENT_TYPE)