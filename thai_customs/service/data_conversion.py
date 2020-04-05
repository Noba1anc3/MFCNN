import numpy as np

class DataConversion():
    def __init__(self):
        pass

    def dict2catalog(self, dict_data, response, label = None):
        dict_res = {}
        commodity = []
        Commodity_key = ''
        if label is None:
            for key in dict_data.keys():
                if len(dict_data[key]) > 0:
                    response.catalog_label = key
                    dict_res = dict_data[key]
        else:
            response.catalog_label = label
            dict_res = dict_data
        for key in dict_res.keys():
            if 'Commodity' in key:
                Commodity_key = key
                commodity = dict_res.pop(Commodity_key)
                break

        self._dict2field(dict_res,response.fields)
        self._list2table(commodity, Commodity_key, response.tables)

    def _list2table(self, list_data, label, tables):
        for dict_data in list_data:
            temp = tables.add()
            temp.table_label = label
            self._dict2field(dict_data, temp.rows)

    def _dict2field(self,dict_data, fields):
        for key in dict_data.keys():
            temp = fields.add()
            temp.field.label = key
            val = dict_data[key]['value']
            if isinstance(val, list):
                for val_i in val:
                    temp.values.append(val_i)
            elif isinstance(val, str):
                temp.values.append(val)
            loc = dict_data[key]['location']
            for loc_i in loc:
                temp1 = temp.file_quad_areas.add()
                temp1.index1.x = loc_i[0][0]
                temp1.index1.y = loc_i[0][1]
                temp1.index2.x = loc_i[1][0]
                temp1.index2.y = loc_i[1][1]
                temp1.index3.x = loc_i[2][0]
                temp1.index3.y = loc_i[2][1]
                temp1.index4.x = loc_i[3][0]
                temp1.index4.y = loc_i[3][1]
                if 'imagename' in dict_data[key]:
                    temp1.file_uri = dict_data[key]['imagename']


    def det_conversion(self, request_det):
        det = []
        box = np.zeros((4, 2), dtype=np.int)
        for det_i in request_det.areaList:
            box[0][0] = det_i.index1.x
            box[0][1] = det_i.index1.y
            box[1][0] = det_i.index2.x
            box[1][1] = det_i.index2.y
            box[2][0] = det_i.index3.x
            box[2][1] = det_i.index3.y
            box[3][0] = det_i.index4.x
            box[3][1] = det_i.index4.y
            det.append(box.copy())
        det = np.array(det)
        return det

    def rec_conversion(self, request_rec):
        rec = []
        for rec_i in request_rec.recognize_result:
            rec.append(rec_i)
        rec = np.array(rec)
        return rec
