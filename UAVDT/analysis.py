# Import the TIDE evaluation toolkit
from tidecv import TIDE

# Import the datasets we want to use
import tidecv.datasets as datasets

bbox_file = '../ddet/UAV_bbox_result_tmp.json'

gt = datasets.COCO('../ddet/data/uav-dataset/UAVDT/UAV-benchmark-M-Val.json')

bbox_results = datasets.COCOResult(bbox_file)

tide = TIDE()

tide.evaluate_range(gt, bbox_results, mode=TIDE.BOX)
tide.summarize()

tide.plot('./UAVDT')