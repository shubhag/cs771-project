# mlt

#Label
Added Labelling application which uses opencv with python to label dataset.

To use the application, first install opencv(most preferably on linux system). Instructions for installing opencv are available on http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html

To use this application, it assumes that there is a example.mp4 file present on the same directory as this file. 
Run > python label.py

Now, first focus your cursor on the popped window.
Then, press f if foreground
            b - background
            q - exit
            
      In case of foreground, just use cursor to draw a rectangle on the object to label. Now, if the object is partially covered then press 'x'.
      Now press key for corresponding class of object.
Currently, classes that can be labelled using the application:
  - Foreground/background - f/b
  - Bus                   - b
  - Pedestrian            - p
  - Motorcycle            - m
  - Tempo                 - t
  - Auto                  - a
  - Cycle                 - c
  - Car                 - f
  - Scooter             - s
  - Exit from foreground frame                  - e

Copy from host to docker container
 cat data.zip |sudo docker exec -i  c45923bbf8c5 sh -c 'cat > /root/keras/d.zip'
 
 Install cuda-7.5 http://tleyden.github.io/blog/2015/11/22/cuda-7-dot-5-on-aws-gpu-instance-running-ubuntu-14-dot-04/
 then docker keras
