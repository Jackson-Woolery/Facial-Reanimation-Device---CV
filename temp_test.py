# Import standard python modules
import time



# import Adafruit IO REST client
from Adafruit_IO import Client

# import Adafruit CircuitPython adafruit_adt7410 library
import adafruit_adt7410

# import Adafruit Blinka
import board
import busio
import digitalio

# Delay between sensor reads, in seconds
DELAY_SECONDS = 30

# Set to your Adafruit IO key.
# Remember, your key is a secret,
# so make sure not to publish it when you publish this code!
ADAFRUIT_IO_KEY = 'aio_xXXz333yR5VRkbBnGamDXbXhkkHW'

# Set to your Adafruit IO username.
# (go to https://accounts.adafruit.com to find your username)
ADAFRUIT_IO_USERNAME = 'SkunkHeadedPunk'

# Create an instance of the REST client
aio = Client(ADAFRUIT_IO_USERNAME, ADAFRUIT_IO_KEY)

# Set up `temperature` feed
pi_temperature = aio.feeds('bridge-b827eb3fd374-sensor-fae925ae3acf.temperature-0')
##all_feeds = aio.feeds(feed=None)
##print(all_feeds)
print(pi_temperature)
pi_temp = aio.receive_previous('bridge-b827eb3fd374-sensor-fae925ae3acf.temperature-0')
print("pi_temp: ", pi_temp)
