#!/usr/bin/python3
#-*- coding: utf-8 -*-
import sys
import os
import threading
import socket
# import urllib.request
import urllib
 
timeout = 4
socket.setdefaulttimeout(timeout)
 
'''
Save image from url data remotely
'''
def download_and_save(url,savename):
    try:
        # data = urllib.request.urlopen(url).read()
        data = urllib.urlopen(url).read()
        fid=open(savename,'w+b')
        fid.write(data)
        print ("download succeed: "+ url)
        fid.close()
    except IOError:
        print ("download failed: "+ url)
 
 
def get_all_image(filename):
    fid = open(filename)
    name = filename.split('\\')[-1]
    name = name[:-4]
    lines = fid.readlines()
    for line in lines:
        line_split = line.split(' ')
        image_id = line_split[0]
        image_url = line_split[1]
        if False == os.path.exists('./vgg_face_dataset/images' + '/' + name):
            os.mkdir('./vgg_face_dataset/images' + '/' + name)
        savefile = './vgg_face_dataset/images' + '/' + name + '/' + image_id + '.jpg'
        if os.path.exists(savefile):
            continue
        #The maxSize of Thread numberr:1000
        print(image_url,savefile)
        while True:
            if(len(threading.enumerate()) < 1000):
                break               
        t = threading.Thread(target=download_and_save,args=(image_url,savefile,))
        t.start()
 
if __name__ == "__main__":
    fileDir = sys.argv[1]
    list = os.listdir(fileDir)
    if len(sys.argv) == 3:
        print ('Download images for subject: %s.'%(sys.argv[2]))
        subject_name = sys.argv[2]
        get_all_image(os.path.join(sys.argv[1],subject_name + '.txt'))
    else:
        for i in range(len(list)):
            get_all_image(os.path.join(sys.argv[1],list[i]))
