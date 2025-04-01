## PULL AND START ULTRALYTICS DOCKER

	# PULL	
	t=ultralytics/ultralytics:latest-jetson-jetpack4
	sudo docker pull $t	
	
	# RUN
	sudo docker run -it --ipc=host \
		 --runtime=nvidia --network host \ 
		-v /home/jetson/code:/ultralytics/code \
		$t


## INFORMATION ABOUT THE COMMAND FLAGS
	sudo docker run -> Runs a new Docker container
	-it 		-> Starts an interactive session (-i keeps STDIN open, -t allocates a TTY)
	--ipc=host	-> Shares the host's IPC namespace (useful for models using shared memory)
	--runtime=nvidia-> Enables the NVIDIA GPU runtime for CUDA access
	-v		-> Mount the local directory (that comes afterwards) to the container
	--network host  -> Share netword with the host (instead of using a default isolated one)


## CREATE A CUSTOM CONTAINER (to save changes and installations)
	sudo docker run -it --ipc=host \
		--runtime=nvidia --network host \
		 --name my_ultralytics  \
		-v /home/jetson/code:/ultralytics/code \
		$t

	sudo docker start -ai my_ultralytics

	sudo docker ps -a

	sudo docker commit <container_id> my_custom_ultralytics

	sudo docker run -it --ipc=host \
		--runtime=nvidia --network host \
		--name my_ultralytics  \
		-v /home/jetson/code:/ultralytics/code \
		my_custom_ultralytics

		
	


