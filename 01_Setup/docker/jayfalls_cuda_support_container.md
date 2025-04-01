## PULL AND START JAYFALLS DOCKER
	
	# PULL	
	t=jayfalls/l4t-20.04:full-cp311
	sudo docker pull $t
	
	# RUN
	sudo docker run -it --ipc=host \
		--runtime=nvidia --network host \
		-v /home/jetson/code:/root/code \ 
		$t


## INSTALL DEPTHAI ON JAYFALLS' DOCKER
	sudo docker run -it --ipc=host --runtime=nvidia --network host \
	    -v /home/jetson/code:/root/code \
	    -v /dev/bus/usb:/dev/bus/usb \
	    --device-cgroup-rule='c 189:* rmw' \
	    --name my_depthai_container \
	    jayfalls/l4t-20.04:full-cp311
	
	# install depthai
		(optional) python3 -m pip install --upgrade pip
		python3 -m pip install depthai --no-deps
		python3 -m pip install --user jupyter

		python3 -c "import depthai; print(depthai.__version__)"
		python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
		python3 -c "import torch; print(torch.cuda.get_device_name(0))"

		python3 -c "import depthai; print(depthai.Device.getAllAvailableDevices())"
		# Should show: 
			[DeviceInfo(name=1.2.3, mxid=18443010C145780E00, 
			X_LINK_UNBOOTED, X_LINK_USB_VSC, X_LINK_MYRIAD_X, X_LINK_SUCCESS)]

		# If error pops:
			[2025-02-27 13:58:36.949] [depthai] [warning] 
			Insufficient permissions to communicate 
			with X_LINK_UNBOOTED device having name "1.2.4". 
			Make sure udev rules are set []

		# Solution:
			echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
			sudo udevadm control --reload-rules && sudo udevadm trigger


	# optional
		apt update && apt install -y libusb-1.0-0

	
	# check mounted usb and video camera (make sure the camera is connected)
		ls /dev/bus/usb

	
	# COMMIT CONTAINER AS A CUSTOM IMAGE
		exit
		sudo docker ps -a
		sudo docker commit my_depthai_container custom_depthai


	# RUN CUSTOM IMAGE AS CONTAINER
		sudo docker run -it --ipc=host --runtime=nvidia --network host \
		    -v /home/jetson/code:/root/code \
		    -v /dev/bus/usb:/dev/bus/usb \
		    --device-cgroup-rule='c 189:* rmw' \
		    -e DISPLAY=$DISPLAY \
		    -v /tmp/.X11-unix:/tmp/.X11-unix \
		    custom_depthai
	
	# on jetson host run:
		xhost +local:root
	(this allows cv2.imshow() to work inside docker

	# (optional) SAVE AND SHARE IMAGE
		sudo docker save -o my_depthai_image.tar my_custom_depthai_image

		