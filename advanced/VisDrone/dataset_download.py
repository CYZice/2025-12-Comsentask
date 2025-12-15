from roboflow import Roboflow

rf = Roboflow(api_key="yZBOkmCH15dzHPc5PUyt")
project = rf.workspace("cyz-2rqmy").project("visdrone-uhzsx-ylc28")
version = project.version(2)
dataset = version.download("yolov11")

data_yaml_path = "data.yaml"
