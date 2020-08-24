from django.conf.urls import url
from core.controller.controller import pos, ner, dependency_parser, sentiment_analysis, tags_extractor, dependency_parser_svg
from core.controller.cyber_bullying_controller import cyber_bullying_controller
from core.controller.sarcasm_detection_controller import sarcasm_detection_controller
from core.controller.tags_recommender_controller import tags_recommender_controller
from core.controller.topic_modelling_controller import topic_modelling_controller
from core.controller.text_summarization_controller import text_summarization_controller
urlpatterns = [
        url(r'^pos/$', pos, name='pos_tagger'),
        url(r'ner/$', ner, name='ner_tagger'),
        url(r'^dependency-parser/$', dependency_parser, name='dependency_parser'),
        url(r'^sentiment/$', sentiment_analysis, name='sentiment_analysis'),
        url(r'^tags/$', tags_extractor, name='tags_extractor'),
        url(r'^cyber_bullying_detection/$', cyber_bullying_controller, name='cyber_bullying_controller'),
        url(r'^sarcasm_detection/$', sarcasm_detection_controller, name='sarcasm_detection_controller'),
        url(r'^tags_recommender/$', tags_recommender_controller, name='tags_recommender_controller'),
	url(r'^topic_modelling/$', topic_modelling_controller, name='topic_modelling_controller'),
	url(r'^text_summarization/$', text_summarization_controller, name='text_summarization_controller')
        ]
