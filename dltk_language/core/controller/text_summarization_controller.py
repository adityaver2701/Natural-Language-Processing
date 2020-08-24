import json
import traceback

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from core.lib.lib import pos_tagger, ner_tagger, parser, sentiment, tags, parser_svg
from core.constants import *
from core.lib.text_summarization import detect

@csrf_exempt
def text_summarization_controller(request):
    try:
        if request.method == "POST":
            print('inside controller')
            params = json.loads(request.body.decode(CHARSET))
            print('outside json loads')
            response = detect(params)
            return HttpResponse(json.dumps(response), content_type=CONTENT_TYPE)
    except Exception as e:
        traceback.print_exc()
        return HttpResponse(e.args, status=500, content_type=CONTENT_TYPE)
