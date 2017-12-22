# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:36:36 2017

@author: BHRS-ZY-PC
"""

import json


ContractName_dict = { "a": u"豆一", 
                     "NI": u"沪镍", 
                     "ZN": u"沪锌", 
                     "l": u"塑料", 
                     "OI": u"菜油",
                     "I": u"铁矿石", 
                     "c": u"玉米", 
                     "J": u"焦炭", 
                     "JM": u"焦煤",
                     "M": u"豆粕", 
                     "AL": u"沪铝", 
                     "CF": u"棉花", 
                     "p": u"棕榈油", 
                     "AU": u"沪金", 
                     "pp": u"聚丙烯",
                     "RB": u"螺纹钢", 
                     "SR": u"白糖", 
                     "JD": u"鸡蛋", 
                     "RM": u"菜粕", 
                     "ZC": u"动力煤", 
                     "CU": u"沪铜", 
                     "TA": u"PTA",
                     "RU": u"橡胶",
                     "AG": u"沪银",
                     "Y" : u"豆油",
                     "TF": u"五年期国债",
                     "T": u"十年期国债",
                     "IF": u"沪深300",
                     "IH": u"上证50",
                     "IC": u"中证500"} 

json_str = json.dumps(ContractName_dict)
new_dict = json.loads(json_str)

with open("CodeParseName.json", "w") as f:
    json.dump(new_dict, f)
