import rosbag
import rospy
import numpy as np
from PIL import Image
import utils
import pickle
#from matplotlib import pyplot as plt

labelPairs = {"/mnts/sdb/Indoor_Data/Data-08-26-22-Time-16-53-18.bag": "/data/zak/rosbag/labeled/heracleia/",
              "/mnts/sdb/Indoor_Data/Data-08-26-22-Time-17-01-06.bag": "/data/zak/rosbag/labeled/heracleia/",
              "/mnts/sdb/Indoor_Data/Data-08-26-22-Time-17-16-38.bag": "/data/zak/rosbag/labeled/mocap/",
              "/mnts/sdb/Indoor_Data/Data-08-26-22-Time-18-13-48.bag": "/data/zak/rosbag/labeled/uc/",
           }

for bagDir, outDir in labelPairs.items():
    bag = rosbag.Bag(bagDir)
    # bagInfo = bag.get_type_and_topic_info()[1]
    # topics = bagInfo.keys()
    # print(topics)

    utils.makeFolder(outDir)
    jointDir = outDir + "joints/"
    imgDir = outDir + "img/"
    frontLaserDir = outDir + "front_laser/"
    utils.makeFolder(jointDir)
    utils.makeFolder(imgDir)
    utils.makeFolder(frontLaserDir)

    for topic, msg, t in bag.read_messages():
        # print(msg.header)
        timeStamp = msg.header.stamp
        # times.append(timeStamp.to_sec())

        if topic == "joints":
            jointDict = {}
            jointDict['name'] = msg.name
            jointDict['position'] = msg.position
            jointDict['velocity'] = msg.velocity
            jointDict['effort']   = msg.effort
            name = jointDir + str(timeStamp) + ".pkl"
            with open(name, "wb") as f:
                pickle.dump(jointDict, f)

        if topic == 'img':
            img = np.frombuffer(msg.data, dtype=np.uint8)
            img = img.reshape(msg.height, msg.width, 3)
            img = Image.fromarray(img)
            img.save(imgDir + str(timeStamp) + ".jpg")

        if topic == "front_laser":
            frontLaserDict = {}
            frontLaserDict["angle_min"] = msg.angle_min
            frontLaserDict["angle_max"] = msg.angle_max
            frontLaserDict["angle_increment"] = msg.angle_increment
            frontLaserDict["time_increment"] = msg.time_increment
            frontLaserDict["scan_time"] = msg.scan_time
            frontLaserDict["range_min"] = msg.range_min
            frontLaserDict["range_max"] = msg.range_max
            frontLaserDict["ranges"] = msg.ranges
            name = frontLaserDir + str(timeStamp) + ".pkl"
            with open(name, "wb") as f:
                pickle.dump(frontLaserDict, f)