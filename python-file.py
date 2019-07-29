
# -*- coding: utf-8 -*-

import sys, faulthandler
# enables the debbigging trace for fatal errors
# faulthandler.enable()
from KNN.knn_sklearn import main as knn_main
from MLL.mll import main as mll_main
from GRNN.grnn import main as grnn_main

# from NER.NERTagger import main as NERMain
# from SA_Module_API_compatible.src.Main import main as SAMain
import json
import re
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


def merge_response(knn_res, mll_res, grnn_res):
	""" combines and converts both responses into json """

	combined_response = {
		"knn":knn_res["knn"],
		"mll":mll_res["mll"],
		"grnn":grnn_res["grnn"]
	}
	
	# convert response to json
	combined_response_json =  json.dumps(combined_response)
	return combined_response_json


def main():
	""" communicates with RAKE and NER module and gets response to combine 
		both repsonses in one """
	try:
		knn_res = knn_main()
		mll_res = mll_main()
		grnn_res = grnn_main()
	except Exception as ex:
		print(ex, file=sys.stderr)
		# print(ex)

	res_json = merge_response(knn_res, mll_res, grnn_res)
	print(res_json)


if __name__ == "__main__":
	main()