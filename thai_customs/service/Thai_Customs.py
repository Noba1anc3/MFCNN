
import time
from logzero import logger

from v_thai_customs.service.bert import info_extract
from v_thai_customs.service.dr2json.handle import Handle
from v_thai_customs.service.mfcn.agents.predict import get_args
from v_thai_customs.service.mfcn.agents.predict import load_model
from v_thai_customs.service.mfcn.agents.predict import mfcn_inference
from v_thai_customs.service.mfcn.agents.predict import json_pros
from v_thai_customs.service.bert.data_process.GEL import GenerateEmbeddingLayer

class Thai_Customs():

    def __init__(self):
        logger.info("--------------------- System Preparing -----------------------")

        self.args = get_args()
        self.auxiliary_path = "bert/auxiliary_768"

        self.handle = Handle(self)
        self.handle._reset_()

        self.GEL = GenerateEmbeddingLayer(self, '.', None)

        self.model_name_inv = 'Invoice_all_for_train_9762_9253.pth'
        self.model_name_pl =  'Packinglist_all_for_train_9869.pth'

        self.inv_keylist = [
            "ignore",
            "INVNo",
            "INVDate",
            "INVTermType",
            "INVShipper",
            "INVConsignee",
            "INVTotalNW",
            "INVTotalGW",
            "INVCurrency",
            "INVTotal",
            "INVTotalQty",
            "INVQtyUom",
            "INVWtUnit",
            "INVCommodity.ItemNo",
            "INVCommodity.BoxNumber",
            "INVCommodity.Desc",
            "INVCommodity.HSCode",
            "INVCommodity.Unit",
            "INVCommodity.Qty",
            "INVCommodity.Price",
            "INVCommodity.PartNumber",
            "INVCommodity.NW",
            "INVCommodity.GW",
            "INVCommodity.COO",
            "INVCommodity.Total",
        ]
        self.pl_keylist = [
            "ignore",
            "PLNo",
            "PLShipper",
            "PLConsignee",
            "PLTotalNW",
            "PLTotalGW",
            "PLWtUnit",
            "PLQtyUom",
            "PLTotal",
            "PLTotalQty",
            "PLCommodity.ItemNo",
            "PLCommodity.PartNumber",
            "PLCommodity.BoxNumber",
            "PLCommodity.Desc",
            "PLCommodity.Unit",
            "PLCommodity.Qty",
            "PLCommodity.Price",
            "PLCommodity.NW",
            "PLCommodity.GW",
            "PLCommodity.COO"
        ]

        self.model_inv, self.model_pl = load_model(self)
        logger.info("--------------------- System Prepared ------------------------\n\n")

    def run(self, inv_pl, image, flip_bboxes, reco_result):

        if inv_pl == 'invoice':
            keylist = self.inv_keylist
            img_type = 'Invoice'
        else:
            keylist = self.pl_keylist
            img_type = 'PackingList'

        start_time = time.time()
        logger.info("<<<<<<<<<<<<<<<<<<<<<<< Image Processing >>>>>>>>>>>>>>>>>>>>>>>")

        dr_to_json = self.handle.det_rec_to_json(image, flip_bboxes, reco_result)
        img, vec128 = info_extract.info_extra(self, img_type, image, dr_to_json, keylist)

        logger.info("<<<<<<<<<<<<<< Image Processed  "
                    + '%.1f'%((time.time() - start_time)*1000) + "ms >>>>>>>>>>>>>>")

        start_time = time.time()

        key_list, value_list, loc_list = mfcn_inference(self, img_type, img, vec128, dr_to_json, keylist)
        logger.info("---- MFCN Inference Completed  "
                    + '%.1f' % ((time.time() - start_time) * 1000) + "ms\n")

        return_json = json_pros(img_type, key_list, value_list, loc_list)
        logger.info("{}\n\n".format(return_json))

        return return_json
