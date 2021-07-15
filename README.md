# Python Movenet

This repository originates from tensorflow/hub and provides python wrapper of Movenet for large datasets pose detection

Place video files under /video/... directory. Video file classified as 'straight' should start with '0\_...' eg. '/video/
/0/0_test.mp4'.

To run all steps use command:
`python ./main.py --skip-vid2pic 0 --skip-movenet 0`

required dependencies can be found in 'setup.sh' file
