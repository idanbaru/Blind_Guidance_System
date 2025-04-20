## Get Thermal Stats from GPU:
	> tegrastats

## Fan control installation:
	> git clone https://github.com/Pyrestone/jetson-fan-ctl.git
	> cd jetson-fan-ctl
	> sudo ./install.sh

## Config fan:
	> sudo nano /etc/automagic-fan/config.json
	> edit wanted sections:
		{
		"FAN_OFF_TEMP":20,
		"FAN_MAX_TEMP":50,
		"UPDATE_INTERVAL":2,
		"MAX_PERF":1
		}
	> sudo service automagic-fan restart

## Checking errors:
	> sudo service automagic-fan status
