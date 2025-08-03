#!/usr/bin/env python

import logging
import os
import json
import traceback
import requests
from pyclowder.extractors import Extractor
import pyclowder.files
import pyclowder.utils
from pyclowder.utils import CheckMessage
from Jim_ColorHistogram_ColorScatterPlot import image_analysis
from optimizer import optimizer_get, optimizer_tell, optimizer_init
from printability import process_image

#TODO, Docker ENV
SCP_WEB_URL_BASE = 'http://host.docker.internal:5000/structural-color-printing/'

class ImageAnalysisExtractor(Extractor):
    def __init__(self):
        Extractor.__init__(self)
        # parse command line and load default logging configuration
        self.setup()
        # setup logging for the exctractor
        logging.getLogger('pyclowder').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)
        self.campaign_id = None
        self.opt = None

    def check_message(self, connector, host, secret_key, resource, parameters):
        logger = logging.getLogger(__name__)
        print(resource["type"])
        if resource["type"] == "metadata":
            # check the type
            if 'metadata' in resource and 'image_analysis' in resource.get('metadata'):
                return CheckMessage.bypass
        return CheckMessage.ignore

    def process_message(self, connector, host, secret_key, resource, parameters):
        # get input file
        inputfile = None
        is_success = True
        try:
            print(f"resource: {resource}")
            file_id = resource['id']
            metadata = resource['metadata']
            campaign_id = metadata['campaign_id']
            cell_id = metadata['cell_id']
            is_skip = bool(metadata['is_skip'])
            rank_run = metadata['rank_run']
            number_prints_trigger_prediction = metadata['number_prints_trigger_prediction']
            accum_h_mu = float(metadata['accum_h_mu'])

            predict_ranges = metadata['predict_ranges']
            my_space = [(float(predict_ranges.get("min_speed")), float(predict_ranges.get("max_speed"))),
                        (float(predict_ranges.get("min_bed_temp")), float(predict_ranges.get("max_bed_temp"))),
                        (float(predict_ranges.get("min_pressure")), float(predict_ranges.get("max_pressure"))),
                        (float(predict_ranges.get("min_zheight")), float(predict_ranges.get("max_zheight")))]

            print(my_space)
            print("campaign_id", campaign_id)
            print("cell_id", cell_id)
            print('rank_run')
            print('number_prints_trigger_prediction')
            inputfile = pyclowder.files.download(connector, host, secret_key, resource['id'])

            printability_score = 100
            h_mu = 0
            h_sig = 0
            v_mu = 0
            v_sig = 0
            s_mu = 0
            s_sig = 0
            if is_skip:
                is_success = False
            else:
                try:
                    H_DIST, h_mu, h_sig, V_DIST, v_mu, v_sig, S_DIST, s_mu, s_sig = image_analysis(inputfile)
                    printability_score = process_image(inputfile)
                except:
                    is_success = False
                    traceback.print_exc()

            content = {
                # 'H_DIST': H_DIST.to_json(),
                       'h_mu': h_mu,
                       'h_sig': h_sig,
                       # 'V_DIST': V_DIST.to_json(),
                       'v_mu': v_mu,
                       'v_sig': v_sig,
                       # 'S_DIST': S_DIST.to_json(),
                       's_mu': s_mu,
                       's_sig': s_sig}
            data = None
            try:
                if is_success:
                    if rank_run == 0:
                        self.campaign_id = campaign_id
                        self.opt = optimizer_init(my_space)
                    if (rank_run +1) % number_prints_trigger_prediction == 0:
                        accum_h_mu += h_mu
                        h_mu = accum_h_mu/number_prints_trigger_prediction
                        combined_objective = h_mu * printability_score
                        PrintSpeed, BedTemp, Pressure, ZHeight = optimizer_get(self.opt)
                        _ = optimizer_tell(self.opt, combined_objective, PrintSpeed, BedTemp, Pressure, ZHeight)
                        data = {"campaign_id": campaign_id, "cell_id": cell_id, "file_id": file_id, 'rank_run': rank_run,
                                "printability_score": printability_score,
                                "cell_color": content,
                                "PrintSpeed": PrintSpeed, "BedTemp": BedTemp, "Pressure": Pressure, "ZHeight": ZHeight,
                                "is_success": True}
                    else:
                        data = {"campaign_id": campaign_id, "cell_id": cell_id, "file_id": file_id, "cell_color": content,
                                 'rank_run': rank_run,
                                "printability_score": printability_score,
                                "is_success": True}
            except:
                is_success = False
                traceback.print_exc()
            if not is_success:
                data = {"campaign_id": campaign_id, "cell_id": cell_id, "file_id": file_id, "cell_color": content,
                        'rank_run': rank_run,
                        "printability_score": printability_score,
                        "is_success": False}
            # store backt to SCP web application
            try:
                url = SCP_WEB_URL_BASE + 'campaign/%s/update_cell_color' % (campaign_id)
                result = requests.post(url, data=json.dumps(data),
                                       headers={'Content-type': 'application/json', 'accept': 'application/json'},
                                       verify=False)
                result.raise_for_status()
            except:
                traceback.print_exc()

        finally:
            if inputfile:
                os.remove(inputfile)



if __name__ == "__main__":
    extractor = ImageAnalysisExtractor()
    extractor.start()