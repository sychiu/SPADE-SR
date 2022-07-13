
import serial
import numpy as np
import time
import glob
import cv2
import collections
import threading
import os


path_to_save_thm = "data/raw_temperature"
path_to_save_rgb = "data/raw_rgb"
rgb_save_shape = (640, 360)


end = False

#VISUALIZATION PARAMETERS (DOENSN'T EFFECT SAVINGS)
upperbound_shift = 1
lowerbound_shift = 1
scaling_windowsize = 5
upper_window = collections.deque(maxlen=scaling_windowsize)
lower_window = collections.deque(maxlen=scaling_windowsize)
med_activate = False
med_filtersize = 3
med_filterwindow = collections.deque(maxlen=med_filtersize)
middleIndex = int(med_filtersize/2)
ma_activate = False
ma_filtersize = 2
ma_filterwindow = collections.deque(maxlen=ma_filtersize)


rgb_buffer = []
temp_buffer = []

record = False
recording = False

display_size = (360,360)
interpolation_size = 64



iron_palette = ["#00000a","#000014","#00001e","#000025","#00002a","#00002e","#000032","#000036","#00003a","#00003e","#000042","#000046","#00004a","#00004f","#000052","#010055","#010057","#020059","#02005c","#03005e","#040061","#040063","#050065","#060067","#070069","#08006b","#09006e","#0a0070","#0b0073","#0c0074","#0d0075","#0d0076","#0e0077","#100078","#120079","#13007b","#15007c","#17007d","#19007e","#1b0080","#1c0081","#1e0083","#200084","#220085","#240086","#260087","#280089","#2a0089","#2c008a","#2e008b","#30008c","#32008d","#34008e","#36008e","#38008f","#390090","#3b0091","#3c0092","#3e0093","#3f0093","#410094","#420095","#440095","#450096","#470096","#490096","#4a0096","#4c0097","#4e0097","#4f0097","#510097","#520098","#540098","#560098","#580099","#5a0099","#5c0099","#5d009a","#5f009a","#61009b","#63009b","#64009b","#66009b","#68009b","#6a009b","#6c009c","#6d009c","#6f009c","#70009c","#71009d","#73009d","#75009d","#77009d","#78009d","#7a009d","#7c009d","#7e009d","#7f009d","#81009d","#83009d","#84009d","#86009d","#87009d","#89009d","#8a009d","#8b009d","#8d009d","#8f009c","#91009c","#93009c","#95009c","#96009b","#98009b","#99009b","#9b009b","#9c009b","#9d009b","#9f009b","#a0009b","#a2009b","#a3009b","#a4009b","#a6009a","#a7009a","#a8009a","#a90099","#aa0099","#ab0099","#ad0099","#ae0198","#af0198","#b00198","#b00198","#b10197","#b20197","#b30196","#b40296","#b50295","#b60295","#b70395","#b80395","#b90495","#ba0495","#ba0494","#bb0593","#bc0593","#bd0593","#be0692","#bf0692","#bf0692","#c00791","#c00791","#c10890","#c10990","#c20a8f","#c30a8e","#c30b8e","#c40c8d","#c50c8c","#c60d8b","#c60e8a","#c70f89","#c81088","#c91187","#ca1286","#ca1385","#cb1385","#cb1484","#cc1582","#cd1681","#ce1780","#ce187e","#cf187c","#cf197b","#d01a79","#d11b78","#d11c76","#d21c75","#d21d74","#d31e72","#d32071","#d4216f","#d4226e","#d5236b","#d52469","#d62567","#d72665","#d82764","#d82862","#d92a60","#da2b5e","#da2c5c","#db2e5a","#db2f57","#dc2f54","#dd3051","#dd314e","#de324a","#de3347","#df3444","#df3541","#df363d","#e0373a","#e03837","#e03933","#e13a30","#e23b2d","#e23c2a","#e33d26","#e33e23","#e43f20","#e4411d","#e4421c","#e5431b","#e54419","#e54518","#e64616","#e74715","#e74814","#e74913","#e84a12","#e84c10","#e84c0f","#e94d0e","#e94d0d","#ea4e0c","#ea4f0c","#eb500b","#eb510a","#eb520a","#eb5309","#ec5409","#ec5608","#ec5708","#ec5808","#ed5907","#ed5a07","#ed5b06","#ee5c06","#ee5c05","#ee5d05","#ee5e05","#ef5f04","#ef6004","#ef6104","#ef6204","#f06303","#f06403","#f06503","#f16603","#f16603","#f16703","#f16803","#f16902","#f16a02","#f16b02","#f16b02","#f26c01","#f26d01","#f26e01","#f36f01","#f37001","#f37101","#f37201","#f47300","#f47400","#f47500","#f47600","#f47700","#f47800","#f47a00","#f57b00","#f57c00","#f57e00","#f57f00","#f68000","#f68100","#f68200","#f78300","#f78400","#f78500","#f78600","#f88700","#f88800","#f88800","#f88900","#f88a00","#f88b00","#f88c00","#f98d00","#f98d00","#f98e00","#f98f00","#f99000","#f99100","#f99200","#f99300","#fa9400","#fa9500","#fa9600","#fb9800","#fb9900","#fb9a00","#fb9c00","#fc9d00","#fc9f00","#fca000","#fca100","#fda200","#fda300","#fda400","#fda600","#fda700","#fda800","#fdaa00","#fdab00","#fdac00","#fdad00","#fdae00","#feaf00","#feb000","#feb100","#feb200","#feb300","#feb400","#feb500","#feb600","#feb800","#feb900","#feb900","#feba00","#febb00","#febc00","#febd00","#febe00","#fec000","#fec100","#fec200","#fec300","#fec400","#fec500","#fec600","#fec700","#fec800","#fec901","#feca01","#feca01","#fecb01","#fecc02","#fecd02","#fece03","#fecf04","#fecf04","#fed005","#fed106","#fed308","#fed409","#fed50a","#fed60a","#fed70b","#fed80c","#fed90d","#ffda0e","#ffda0e","#ffdb10","#ffdc12","#ffdc14","#ffdd16","#ffde19","#ffde1b","#ffdf1e","#ffe020","#ffe122","#ffe224","#ffe226","#ffe328","#ffe42b","#ffe42e","#ffe531","#ffe635","#ffe638","#ffe73c","#ffe83f","#ffe943","#ffea46","#ffeb49","#ffeb4d","#ffec50","#ffed54","#ffee57","#ffee5b","#ffee5f","#ffef63","#ffef67","#fff06a","#fff06e","#fff172","#fff177","#fff17b","#fff280","#fff285","#fff28a","#fff38e","#fff492","#fff496","#fff49a","#fff59e","#fff5a2","#fff5a6","#fff6aa","#fff6af","#fff7b3","#fff7b6","#fff8ba","#fff8bd","#fff8c1","#fff8c4","#fff9c7","#fff9ca","#fff9cd","#fffad1","#fffad4","#fffbd8","#fffcdb","#fffcdf","#fffde2","#fffde5","#fffde8","#fffeeb","#fffeee","#fffef1","#fffef4","#fffff6"]
for u in range(len(iron_palette)):
    color = tuple(int(iron_palette[u].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    iron_palette[u] = (color[2],color[1],color[0])



def inputHandle():
    global med_filtersize
    global ma_filtersize
    global ma_filterwindow
    global med_filterwindow
    global middleIndex
    global end
    global upperbound_shift
    global lowerbound_shift
    global record
    while(end!=True):
        try:
            chars = input()
            if(chars == 'end'):
                end = True
            elif(chars=='rcd'):
                print("COUNTING DOWN IN 4")
                time.sleep(4)
                print("START RECORDING")
                record = True
            elif(chars=='stop'):
                record = False
            else:
                print("INPUT ERROR")

        except:
            print("INPUT ERROR")
    

def median_filter(med_filterwindow):
    med_filterwindow = np.sort(np.array(med_filterwindow), axis=0)
    return med_filterwindow[middleIndex]


def ma(ma_filterwindow):
    return np.mean(ma_filterwindow,axis=0)

print("GREETINGS...")
while(1):
    try:
        ser = serial.Serial()
        ser.baudrate = 115200
        ser.port = portName
        ser.timeout = 2
        ser.open() # check port avalibility
        break
    
    except Exception as e:
        print('')
        print(e)
        print("Oops, Something wrong with the serial connection !")
        print("Select the port to connect:")
        portList = glob.glob('/dev/tty.*')
        for i, o in enumerate(portList):
            print("[%d]%s" % (i, o))
        choose = input()
        try:
            portName = portList[int(choose)]
        except:
            pass


print("HURRAY!! OFF WE GO")
cap = cv2.VideoCapture(0)
ser.flush()
ser.reset_input_buffer() #Get rid of initial incomplete data

t = 0
thread = threading.Thread(target = inputHandle)
thread.start()
ser.reset_input_buffer() #Get rid of initial incomplete data
clearOut = ser.readline() 
clearOut = None

while(end!=True):

    if ser.in_waiting > 390:
        #READ
        data = ser.readline().decode()[:-2]
        if ser.in_waiting <= 390:
            raw_temperatures = data.split(',')
            device_temp = float(raw_temperatures.pop())
            raw_temperatures = np.array(raw_temperatures, dtype=np.float32)
            temperatures = np.copy(raw_temperatures)
    
            #FILTERING
            if(med_activate==True):
                med_filterwindow.append(temperatures)
                if(len(med_filterwindow)>=med_filtersize):
                    temperatures = median_filter(med_filterwindow)
    
            if(ma_activate==True):
                ma_filterwindow.append(temperatures)
                if(len(ma_filterwindow)>=ma_filtersize):
                    temperatures = ma(ma_filterwindow)
    
    
            print("MAX: %.2f, MIN: %.2f, DEVICE: %.2f, FPS: %.1f" % (np.max(temperatures), np.min(temperatures), device_temp, 1/(time.time()-t)))
            t = time.time()
            
            #SCALING
            upper_window.append(np.max(temperatures))
            lower_window.append(np.min(temperatures))
    
            if(len(upper_window)>=scaling_windowsize):
                upperbound = np.mean(upper_window)+upperbound_shift
                lowerbound = np.mean(lower_window)+lowerbound_shift
            
                temperatures = (temperatures-lowerbound)/(upperbound-lowerbound+1.e-8)
            temperatures = np.maximum(temperatures,0.0)
            temperatures = np.minimum(temperatures,1.0)
    
            #VISUALIZATION
            thermal_image = np.zeros((8,8,3), np.uint8) 
            for y in range(8):
                for x in range(8):
                    cv2.rectangle(thermal_image, (int(y), int(x)), (int((y+1)), int((x+1))), iron_palette[int(temperatures[(7-y)*8+(x)]*432)], -1)
    
            resized_thermal_image = cv2.resize(thermal_image, (interpolation_size, interpolation_size), interpolation=cv2.INTER_CUBIC)
            thermal_display = cv2.resize(resized_thermal_image, (display_size[0],display_size[1]), interpolation=cv2.INTER_NEAREST)
            
            #JUST COARSE ALIGNMENT, PLEASE DO THE ALIGMENT AFTERWARDS
            #thermal_display = thermal_display[int(display_size[1]*0.085):int(display_size[1]*0.75),:]
            cv2.imshow('thermal', thermal_display)
            
        
            ret, frame = cap.read()
            frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            #JUST COARSE ALIGNMENT, PLEASE DO THE ALIGMENT AFTERWARDS
            #rgbDisplay =  frame[:,10:1630]
            rgbDisplay = cv2.resize(rgbDisplay, (thermal_display.shape[1],thermal_display.shape[0]), interpolation=cv2.INTER_AREA)
            cv2.imshow("rgb", rgbDisplay)
            overlay = cv2.addWeighted(rgbDisplay,0.4,thermal_display,0.5,30)
            cv2.imshow("overlay", overlay)
            cv2.waitKey(1)

            if(record==True):#record data
                recording = True
                rgb_buffer.append(cv2.resize(frame, rgb_save_shape, interpolation=cv2.INTER_AREA))
                temp_buffer.append(raw_temperatures)
            
            elif(recording==True and record==False): #save
                print('SAVING.....')
                recording = False
                save_id = str(int(time.time()*1000))
                np.save(os.path.join(path_to_save_thm, save_id),temp_buffer)
                
                rgb_path = os.path.join(path_to_save_rgb, save_id)
                os.mkdir(rgb_path)
                for i in range(len(rgb_buffer)):
                    cv2.imwrite(os.path.join(rgb_path,str(i)+'.png'),rgb_buffer[i])
    
                rgb_buffer[:] = []
                temp_buffer[:] = []
                print('SAVED!')
        else:
            print(ser.in_waiting)
            
ser.reset_input_buffer() #Get rid of initial incomplete data
clearOut = ser.readline() 
clearOut = None
ser.close()
print('THANK YOU')








