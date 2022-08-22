import rosbag
import rospy
import numpy as np
from PIL import Image
import utils
import pickle
#from matplotlib import pyplot as plt

bagDir = "/mnt/Indoor_Data/Data-08-12-22-Time-17-32-56.bag"
outDir = "/data/zak/rosbag/erb/"
utils.makeFolder(outDir)
bag = rosbag.Bag(bagDir)
# bagInfo = bag.get_type_and_topic_info()[1]
# topics = bagInfo.keys()

for topic, msg, t in bag.read_messages():
    timeStamp = msg.header.stamp
    if topic == "joints":
        jointDict['name'] = msg.name
        jointDict['position'] = msg.position
        jointDict['velocity'] = msg.velocity
        jointDict['effort']   = msg.effort
        jointDir = outDir + topic + "/"
        makeFolder(jointDir)
        with open(jointDir + str(timeStamp) + ".pkl", "wb") as f:
            pickle.dump(jointDict, f)
            
    if topic == 'img':
        img = np.frombuffer(msg.data, dtype=np.uint8)
        img = img.reshape(msg.height, msg.width, 3)
        img = Image.fromarray(img)
        imgDir = outDir + topic + "/"
        utils.makeFolder(imgDir)
        img.save(imgDir + str(timeStamp) + ".jpg")
   