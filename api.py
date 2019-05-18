import logging
from functools import wraps

import cv2
from flask import Flask, request, Blueprint
from flask_cors import CORS
from flask_restplus import Api, Resource, fields
from werkzeug.contrib.fixers import ProxyFix

from logo_detector import *

application = Flask(__name__)
blueprint = Blueprint('api', __name__, url_prefix='/api')

CORS(application)
gunicorn_error_logger = logging.getLogger('gunicorn.error')
application.logger.handlers.extend(gunicorn_error_logger.handlers)
application.logger.setLevel(logging.DEBUG)
application.logger.debug('this will show in the log')
application.wsgi_app = ProxyFix(application.wsgi_app)

authorizations = {
    'apikey': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'X-API-KEY'
    }
}

# authorizations=authorizations to make authorization
api = Api(blueprint, doc='/documentation', title='PEPSI LOGO DETECTOR',
          description='API documentation for detecting PEPSI logo in images',
          default='LogoDetector', default_label='namespace covering api calls to detect PEPSI logo in images')
application.register_blueprint(blueprint)


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'X-API-KEY' in request.headers:
            token = request.headers['X-API-KEY']
        if not token:
            return {'message': 'Token X-API-KEY is missing'}, 401
        if token != 'mytoken':
            return {'message': 'Your token is wrong!'}, 401
        return f(*args, **kwargs)

    return decorated


image_model = api.model('image', {'encoded image': fields.String('image')})
response_model = api.model('detections', {'detectedBoxes': fields.Raw([])})


## App error handler

@api.route('/detect_logo')
class ModelQuery(Resource):

    @api.doc(responses={400: 'Unsuccessful', 406: 'Not Acceptable', 500: 'Internal Server Error/Model not ready yet'})
    @api.expect(image_model)
    @api.marshal_with(response_model)
    def post(self):
        img_parsed = request.data
        img = cv2.imdecode(np.fromstring(img_parsed, dtype=np.uint8), cv2.IMREAD_COLOR)

        # Processing image: Detecting logo
        boxes = []
        print('Image processing...')
        boxes_nms = detect_logo(img)
        print('Done!')
        if boxes_nms is not None:
            for i in range(boxes_nms.shape[0]):
                rect = boxes_nms[i, :].tolist()
                boxes += [rect]
        else:
            boxes_nms = []

        response = {'detectedBoxes': boxes}

        return response, 201


if __name__ == '__main__':
    application.run(debug=True, use_reloader=False)
