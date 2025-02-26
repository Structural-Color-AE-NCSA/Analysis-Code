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
        try:
            file_id = resource['id']
            metadata = resource['metadata']
            campaign_id = metadata['campaign_id']
            cell_id = metadata['cell_id']
            print("campaign_id", campaign_id)
            print("cell_id", cell_id)
            inputfile = pyclowder.files.download(connector, host, secret_key, resource['id'])
            H_DIST, h_mu, h_sig, V_DIST, v_mu, v_sig, S_DIST, s_mu, s_sig = image_analysis(inputfile)
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
            # format the conent as a metadata
            # metadata = self.get_metadata(content, "file", parameters['id'], host)

            # upload metadata
            # pyclowder.files.upload_metadata(connector, host, secret_key, parameters['id'], metadata)

            if self.campaign_id is None or self.campaign_id != campaign_id:
                self.campaign_id = campaign_id
                self.opt = optimizer_init()

            PrintSpeed, BedTemp, Pressure, ZHeight = optimizer_get(self.opt)
            _ = optimizer_tell(self.opt, h_mu, PrintSpeed, BedTemp, Pressure, ZHeight)
            # store backt tp SCP web application
            try:
                url = SCP_WEB_URL_BASE + 'campaign/%s/update_cell_color' % (campaign_id)
                result = requests.post(url, data=json.dumps(
                    {"campaign_id": campaign_id, "cell_id": cell_id, "file_id": file_id, "cell_color": content,
                     "PrintSpeed": PrintSpeed, "BedTemp": BedTemp, "Pressure": Pressure, "ZHeight": ZHeight}),
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
